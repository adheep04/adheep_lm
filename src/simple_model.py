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

    d_latent_rope: int = 32
    rope_base: float = 10_000.0
    rope_factor: float = 4.0
    beta_fast: float = 32.0
    beta_slow: float = 2.0

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
        self.d_nope_head = args.d_nope_head
        self.d_rope_head = args.d_rope_head
        self.d_head_qk = self.d_rope_head + self.d_nope_head
        self.wq_a = nn.Linear(self.d_model, self.d_latent_q) 
        self.q_norm = RMSNorm(self.d_latent_q)
        self.wq_b = nn.Linear(self.d_latent_q, self.n_heads * self.d_head_qk)
        # down projection has a section for the kv latent cache and the key_rope
        # projected input is split during the forward pass into the two parts
        self.wkv_a = nn.Linear(self.d_model, self.d_latent_kv + self.d_rope_head)
        self.kv_norm = RMSNorm(self.d_latent_kv)
        self.wkv_b = nn.Linear(self.d_latent_kv, self.n_heads * (self.d_nope_head + self.d_head_v))
        self.wo = nn.Linear(self.n_heads * self.d_head_v, self.d_model)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.d_latent_kv), persistent=False)
        self.register_buffer("key_rope_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.d_rope_head), persistent=False)
        # scalable softmax is a drop in replacement for standard softmax that helps with long context: https://arxiv.org/pdf/2501.19399
        self.ssmax = ScalableSoftmax()
        self.sqrt_d_model = self.d_head_qk ** -0.5
        self.training = args.training

    def forward(self, h: torch.Tensor, start_pos: int, rot_factors: torch.Tensor, mask: Optional[torch.Tensor], value_skip: Optional[torch.Tensor]):
        # get input dims
        batch_size, seq_len, _ = h.size()
        # get end position
        end_pos = start_pos + seq_len

        # input -> down proj -> norm -> up proj (parts of head have different uses)
        # (b, sl, dm) -> (b, sl, q) -> (b, sl, h * qkh)
        query = self.wq_b(self.q_norm(self.wq_a(h)))
        # reshape query: (b, sl, dm) -> (b, sl, q) -> (b, sl, h, qkh)
        query = query.view(batch_size, seq_len, self.n_heads, self.d_head_qk)
        # split query into diff uses; no pos and pos: (b, sl, h, qkh) -> (b, sl, h, nh), (b, sl, h, rh)
        query_nope, query_rope = torch.split(query, [self.d_nope_head, self.d_rope_head], dim=-1)
        # apply rope to query_rope
        query_rope = apply_rotary_emb(query_rope, rot_factors)

        # input -> latent kv
        # (b, sl, dm) -> (b, sl, kv + rh)
        kv_latent = self.wkv_a(h)
        # split kv into kv_nope and key_rope: (b, sl, kv + rh) -> (b, sl, kv), (b, sl, rh)
        kv, key_rope = torch.split(kv_latent, [self.d_latent_kv, self.d_rope_head], dim=-1)
        # apply rope
        key_rope = apply_rotary_emb(key_rope.unsqueeze(2), rot_factors)
        # grab weights of shape (kv, (nh + vh), h)

        wkv_b = self.wkv_b.weight
        # reshape weights to shape (h, (nh + vh), kv)
        wkv_b = wkv_b.view(self.n_heads, -1, self.d_latent_kv)
        # proj query_nope to kv space: (b, sl, h, nh) * (h, nh, kv) -> (b, sl, h, kv)
        query_nope_kv = torch.einsum('bshn,hnc->bshc', query_nope, wkv_b[:, :self.d_nope_head])
        # kv_cache: (b, tl, kv)
        self.kv_cache[:batch_size, start_pos:end_pos] = self.kv_norm(kv)
        # kv_cache: (b, tl, rh)
        self.key_rope_cache[:batch_size, start_pos:end_pos] = key_rope.squeeze(2)

        # scores = qk_nope_score + qk_rope_score
        scores = (
            # nope attn score: (b, sl, h, kv) * (b, tl, kv) -> (b, sl, h, tl)
            torch.einsum('bshc,btc->bsht', query_nope_kv, self.kv_cache[:batch_size, :end_pos]) +
            # rope attn score: (b, sl, h, rh) * (b, tl, rh) -> (b, sl, h, tl)
            torch.einsum('bshr,btr->bsht', query_rope, self.key_rope_cache[:batch_size, :end_pos])
            # scale for softmax stability
        ) * self.sqrt_d_model
        if mask is not None:

            scores += mask.unsqueeze(1)
        scores = self.ssmax(scores, dim=-1).type_as(h)
        # apply scores to kv_cache latent compression
        v_latent = torch.einsum('bsht,btc->bshc', scores, self.kv_cache[:batch_size, :end_pos])
        # apply value-split of the kv_b weight to latent kv_cache that had score applied
        v = torch.einsum('bshc,hvc->bshv', v_latent, wkv_b[:, -self.d_head_v:])
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
        self.transformer = nn.ModuleDict(dict(
            tok_embed = nn.Embedding(self.vocab_size, args.d_model),
            layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])


        ))

        self.prenorm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.lm_head = nn.Linear(args.d_model, self.vocab_size, bias=False)

        self.head_dim = self.args.d_model // self.args.n_heads
        self.freqs_complex = precompute_rot_factors(self.args.d_model // self.args.n_heads, self.args.max_seq_len * 2, self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (batch_size, seq_len)
        batch_size, seq_len = tokens.shape

        h = self.tok_embeddings(tokens)

        freqs = None

        for layer in self.layers:
            h = layer(h, start_pos, freqs)
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

def precompute_rot_factors(args: ModelArgs) -> torch.Tensor:
    factor = args.rope_factor
    base = args.rope_base
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow

     

    return torch.zeros(10, 10)

def apply_rotary_emb(h: torch.Tensor, rot_factors: torch.Tensor) -> torch.Tensor:
    return h

if __name__ == '__main__':
    tens = torch.randn(0)
    ssmax_ = ScalableSoftmax()
    print(f" ssmax: {ssmax_(tens, dim=0)}")
    print(f" softmax: {tens.softmax(dim=0)}")
    print(f" original: {tens}")
