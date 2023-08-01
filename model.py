from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import pax


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs)
    freqs_cis = jnp.exp(1j * freqs)  # use Euler's formula to create complex numbers
    return freqs_cis


def reshape_for_broadcast(freqs_cis: jnp.ndarray, x: jnp.ndarray):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.reshape(*shape)


def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    # Convert tensors to complex numbers and reshape
    xq_ = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    # Manually creating complex numbers
    xq_ = jax.lax.complex(xq_[..., 0], xq_[..., 1])
    xk_ = jax.lax.complex(xk_[..., 0], xk_[..., 1])

    # Prepare the freqs_cis tensor for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # Perform complex number multiplication
    xq_out = xq_ * freqs_cis
    xk_out = xk_ * freqs_cis

    # Separate real and imaginary parts and reshape the output
    xq_out = jnp.stack([jnp.real(xq_out), jnp.imag(xq_out)], axis=-1).reshape(*xq.shape)
    xk_out = jnp.stack([jnp.real(xk_out), jnp.imag(xk_out)], axis=-1).reshape(*xk.shape)

    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


def repeat_kv(x, n_rep: int):
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        x = jnp.tile(x[:, :, :, None, :], (1, 1, 1, n_rep, 1))
        x = x.reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        return x


class RMSNorm(pax.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = jnp.ones([dim], dtype=jnp.float32)

    def _norm(self, x):
        return x * jax.lax.rsqrt(jnp.power(x, 2).mean(-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(jnp.float32)).astype(x.dtype)
        return output * self.weight


class Linear(pax.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = jnp.zeros((out_dim, in_dim), dtype=jnp.float32)

    def __call__(self, x):
        return jnp.dot(x, self.weight.T)


class Attention(pax.Module):
    def __init__(self, args):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = Linear(args.dim, args.n_heads * self.head_dim)
        self.wk = Linear(args.dim, self.n_kv_heads * self.head_dim)
        self.wv = Linear(args.dim, self.n_kv_heads * self.head_dim)
        self.wo = Linear(args.n_heads * self.head_dim, args.dim)
        self.cache_k = jnp.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ),
            dtype=jnp.float32,
        )
        self.cache_v = jnp.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ),
            dtype=jnp.float32,
        )

    def __call__(self, x, start_pos, freqs_cis, mask=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.reshape(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = jnp.asarray(self.cache_k, xq.dtype)
        self.cache_v = jnp.asarray(self.cache_v, xq.dtype)

        self.cache_k = jax.lax.dynamic_update_slice(
            self.cache_k, xk, (0, start_pos, 0, 0)
        )
        self.cache_v = jax.lax.dynamic_update_slice(
            self.cache_v, xv, (0, start_pos, 0, 0)
        )

        # keys = self.cache_k[:bsz]
        # keys = jax.lax.dynamic_slice_in_dim(keys, 0, start_pos + seqlen, axis=1)
        # values = self.cache_v[:bsz]
        # values = jax.lax.dynamic_slice_in_dim(values, 0, start_pos + seqlen, axis=1)
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = jnp.swapaxes(xq, 1, 2)
        keys = jnp.swapaxes(keys, 1, 2)
        values = jnp.swapaxes(values, 1, 2)
        scores = jnp.matmul(xq, jnp.swapaxes(keys, 2, 3)) / jnp.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(xq.dtype)
        output = jnp.matmul(scores, values)
        output = jnp.swapaxes(output, 1, 2).reshape(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(pax.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(dim, hidden_dim)
        self.w2 = Linear(hidden_dim, dim)
        self.w3 = Linear(dim, hidden_dim)

    def __call__(self, x):
        return self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(pax.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(self, x, start_pos, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Embedding(pax.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.weight = jnp.zeros((vocab_size, embed_dim), dtype=jnp.float32)

    def __call__(self, x):
        return self.weight[(x,)]


class Transformer(pax.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = Embedding(params.vocab_size, params.dim)

        self.layers = []
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(params.dim, params.vocab_size)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def __call__(self, tokens, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = jax.lax.dynamic_slice_in_dim(
            self.freqs_cis, start_pos, seqlen, axis=0
        )
        mask = None
        if seqlen > 1:
            mask = jnp.full((1, 1, seqlen, seqlen), -jnp.inf)
            # mask = jnp.triu(mask, start_pos + 1).astype(h.dtype)
            mask = jnp.triu(mask, 1).astype(h.dtype)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).astype(jnp.float32)
        return output


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
