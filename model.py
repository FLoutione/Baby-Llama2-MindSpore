import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import mindspore
from mindspore import nn, ops
from mindspore.common.initializer import initializer, Normal

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0


class RMSNorm(nn.Cell):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = mindspore.Parameter(ops.ones(dim))

    def _norm(self, x):
        return x * ops.rsqrt(x.pow(2).mean(-1, keep_dims=True) + self.eps)

    def construct(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (ops.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = ops.arange(end)  # type: ignore
    freqs = ops.outer(t, freqs).float()  # type: ignore
    freqs_cos = ops.cos(freqs)  # real part
    freqs_sin = ops.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: mindspore.Tensor, x: mindspore.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: mindspore.Tensor,
    xk: mindspore.Tensor,
    freqs_cos: mindspore.Tensor,
    freqs_sin: mindspore.Tensor
) -> Tuple[mindspore.Tensor, mindspore.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = ops.stack([xq_out_r, xq_out_i], axis=-1).flatten(3)
    xk_out = ops.stack([xk_out_r, xk_out_i], axis=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: mindspore.Tensor, n_rep: int) -> mindspore.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .broadcast_to([bs, slen, n_kv_heads, n_rep, head_dim])
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Cell):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Dense(args.dim, args.n_heads * self.head_dim, has_bias=False)
        self.wk = nn.Dense(args.dim, self.n_kv_heads * self.head_dim, has_bias=False)
        self.wv = nn.Dense(args.dim, self.n_kv_heads * self.head_dim, has_bias=False)
        self.wo = nn.Dense(args.n_heads * self.head_dim, args.dim, has_bias=False)
        self.attn_dropout = nn.Dropout(p=args.dropout)
        self.resid_dropout = nn.Dropout(p=args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation?
        print("WARNING: using slow attention. Flash Attention not supported")
        mask = ops.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = ops.triu(mask, diagonal=1)
        self.mask = mask

    def construct(
        self,
        x: mindspore.Tensor,
        freqs_cos: mindspore.Tensor,
        freqs_sin: mindspore.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.swapaxes(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.swapaxes(1, 2)
        xv = xv.swapaxes(1, 2)

        # manual implementation
        scores = ops.matmul(xq, xk.swapaxes(2, 3)) / math.sqrt(self.head_dim)
        assert hasattr(self, 'mask')
        scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = ops.softmax(scores.float(), axis=-1).type_as(xq)
        scores = self.attn_dropout(scores)
        output = ops.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.swapaxes(1, 2).view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Cell):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Dense(dim, hidden_dim, has_bias=False)
        self.w2 = nn.Dense(hidden_dim, dim, has_bias=False)
        self.w3 = nn.Dense(dim, hidden_dim, has_bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, x):
        return self.dropout(self.w2(ops.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Cell):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def construct(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.construct(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.construct(self.ffn_norm(h))
        return out


class Transformer(nn.Cell):
    last_loss: Optional[mindspore.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(p=params.dropout)
        self.layers = nn.CellList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Dense(params.dim, params.vocab_size, has_bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.embedding_table = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.parameters_and_names():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                p.set_data(initializer(Normal((0.02 / math.sqrt(2 * params.n_layers))),
                                              p.shape, p.dtype))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, cell):
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Normal(0.02),
                                            cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = initializer(Normal(0.02),
                                    cell.weight.shape, cell.weight.dtype)
            cell.weight.set_data(weight)

    def construct(self, tokens: mindspore.Tensor, targets: Optional[mindspore.Tensor] = None) -> mindspore.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            self.last_loss = ops.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, beta1, beta2):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.parameters_and_names()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.ndimension() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.ndimension() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(nn.AdamWeightDecay(eps=1e-8, weight_decay=0.01)).parameters
        use_fused = fused_available #and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = nn.AdamWeightDecay(optim_groups, learning_rate=learning_rate, beta1=beta1, beta2=beta2, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters_dict())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    #@torch.inference_mode()
    #@torch.no_grad()
    def generate(self, idx, eos, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.shape[1] <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = ops.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = ops.topk(logits, min(top_k, logits.shape[-1]))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = ops.softmax(logits, axis=-1)
                idx_next = ops.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = ops.cat((idx, idx_next), axis=1)
            if idx_next==eos:
                break

        return idx

    def export(self, filepath='model.bin'):
        """export the model weights in fp32 into .bin file to be read from C"""
        f = open(filepath, 'wb')

        def serialize(t):
            d = t.view(-1).asnumpy().astype(np.float32)
            b = struct.pack(f'{len(d)}f', *d)
            f.write(b)

        # first write out the header
        hidden_dim = self.layers[0].feed_forward.w1.weight.shape[0]
        p = self.params
        n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
        header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                       n_kv_heads, p.vocab_size, p.max_seq_len)
        f.write(header)

        # next write out the embedding weights
        serialize(self.tok_embeddings.embedding_table)

        # now all the layers
        # attention weights
        for layer in self.layers:
            serialize(layer.attention_norm.weight)
        for layer in self.layers:
            serialize(layer.attention.wq.weight)
        for layer in self.layers:
            serialize(layer.attention.wk.weight)
        for layer in self.layers:
            serialize(layer.attention.wv.weight)
        for layer in self.layers:
            serialize(layer.attention.wo.weight)
        # ffn weights
        for layer in self.layers:
            serialize(layer.ffn_norm.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w1.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w2.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w3.weight)
        # final rmsnorm
        serialize(self.norm.weight)
        # note: no need to write final classifier weights due to weight sharing
        # freqs_cis
        serialize(self.freqs_cos[:p.max_seq_len])
        serialize(self.freqs_sin[:p.max_seq_len])

        # write to binary file
        f.close()
        print(f"wrote {filepath}")