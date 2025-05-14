import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from scalable_softmax import ScalableSoftmax

@dataclass
class ModelArgs:
    d_model: int = 1024 
    n_heads: int = 16 
    d_head_v: int = 64 
    n_layers: int = 36 

    d_latent_kv: int = 128
    d_latent_q: int = 256
    d_nope_head: int = 10
    d_rope_head: int = 10
    d_head_qk: int = d_rope_head + d_nope_head

    vocab_size: int = 32_000 
    d_ffn: int = 4096
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    max_batch_size: int = 16

    device: str = "cuda"
    training: bool = True
    dropout: float = 0.06


class MultiheadLatentAttn(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.d_head_v = args.d_head_v
        self.n_layers = args.n_layers
        self.d_latent_kv = args.d_latent_kv
        self.d_latent_q = args.d_latent_q
        self.d_head_qk = args.d_head_qk
        self.wq_a = nn.Linear(self.d_model, self.d_latent_q) 
        self.q_norm = RMSNorm(self.d_latent_q)
        self.wq_b = nn.Linear(self.d_latent_q, self.n_heads * self.d_head_qk)
        self.wkv_a = nn.Linear(self.d_model, self.d_latent_kv)
        self.kv_norm = RMSNorm(self.d_latent_kv)
        self.wkv_b = nn.Linear(self.d_latent_kv, self.n_heads * (self.d_head_qk + self.d_head_v))
        self.wo = nn.Linear(self.n_heads * self.d_head_v, self.d_model)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.d_latent_kv), persistent=False)
        self.register_buffer("key_rope_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.d_rope_head), persistent=False)
        self.softpick = ScalableSoftmax()
        self.sqrt_d_model = self.d_head_qk ** -0.5
        self.training = args.training

    def forward(self, h: torch.Tensor, value_skip: Optional[torch.Tensor], start_pos: int, mask: Optional[torch.Tensor]):
        # get input dims
        batch_size, seq_len, _ = h.size()
        # get end position
        end_pos = start_pos + seq_len
        # input -> down proj -> norm -> up proj (parts of head have different uses)
        # (b, sl, dm) -> (b, sl, q) -> (b, sl, h * qkh)
        query = self.wq_b(self.q_norm(self.wq_a(h)))
        # reshape query: (b, sl, dm) -> (b, sl, q) -> (b, sl, h, qkh)
        query = query.view(batch_size, seq_len, self.n_heads, self.d_head_qk)

        # input -> latent kv
        # (b, sl, dm) -> (b, sl, kv)
        kv = self.wkv_a(h)

        # (kv, h * (qkh + vh))
        wkv_b = self.wkv_b.weight
        # reshape weights to shape (h, (qkh + vh), kv)
        wkv_b = wkv_b.view(self.n_heads, -1, self.d_latent_kv)
        # proj query to kv space: (b, sl, h, qkh) * (h, qkh, kv) -> (b, sl, h, kv)
        query_kv = torch.einsum('bshq,hqc->bshc', query, wkv_b[:, :self.d_head_qk])
        # kv_cache: (b, tl, kv)
        self.kv_cache[:batch_size, start_pos:end_pos] = self.kv_norm(kv)

        # attn score: (b, sl, h, kv) * (b, tl, kv) -> (b, sl, h, tl) multiply by inverse sqrt of dm for softmax stability
        scores = (torch.einsum('bshc,btc->bsht', query_kv, self.kv_cache[:batch_size, :end_pos])) * self.sqrt_d_model
        if mask is not None:
            scores += mask.unsqueeze(1)

        scores = self.ssmax(scores, dim=-1).type_as(h)
        # (b, sl, h, tl) * (b, tl, kv) -> (b, sl, h, kv)
        v_latent = torch.einsum('bsht,btc->bshc', scores, self.kv_cache[:batch_size, :end_pos])
        # (b, sl, h, kv) * (h, vh, kv) -> (b, sl, h, vh)
        v = torch.einsum('bshc,hvc->bshv', v_latent, wkv_b[:, -self.d_head_v:])
        # (b, sl, h, vh) -> (b, sl, h*vh) * (h*vh, dm) -> (b, sl, dm)
        o = self.wo(v.flatten(2))
        return o


class LanguageModel(nn.Module):
    device: str = 'cuda'
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size != -1, "vocab size needs to be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embed = nn.Embedding(self.vocab_size, args.d_model),
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.prenorm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.lm_head = nn.Linear(args.d_model, self.vocab_size, bias=False)
        self.head_dim = self.args.d_model // self.args.n_heads

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (batch_size, seq_len)
        batch_size, seq_len = tokens.shape

        h = self.tok_embeddings(tokens)
        # value skip if the layer number is in the last quarter of layers (e.g. the last 9 in 36 total layers)
        v_skip_layers = [i for i in range(self.n_layers) if i >= 0.75 * self.n_layers]

        # run through first transformer layer to get value skip
        h, value_skip = self.layers[0](h, start_pos, self.rot_factors)
        # start from the next layer
        for i, layer in enumerate(self.layers[1:], start=1): 
            # add sparse value residual skip connections
            if i not in v_skip_layers:
                h = layer(h, start_pos, self.rot_factors)
            else:
                h = layer(h, value_skip, start_pos, self.rot_factors)
        h = self.prenorm(h)
        return self.lm_head(h)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.attn = MultiheadLatentAttn()
        self.mlp_norm = MultiheadLatentAttn()
        self.mlp = MultiheadLatentAttn()

    def forward(self, h: torch.Tensor):
        h = h + self.attn(self.attn_norm(h)) 
        h = h + self.mlp(self.mlp_norm(h)) 

if __name__ == '__main__':
    tens = torch.randn(0)
    ssmax_ = ScalableSoftmax()
    print(f" ssmax: {ssmax_(tens, dim=0)}")
    print(f" softmax: {tens.softmax(dim=0)}")
    print(f" original: {tens}")
