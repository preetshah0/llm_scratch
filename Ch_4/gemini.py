import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_CONFIG = {
    "vocab_size": 256000,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "activation": "SwiGLU",
}

GEMINI_MODEL_CONFIGS = {
    "gemini-micro": {
        "emb_dim": 128,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 4,
        "mlp_dim": 341,
    },
    "gemini-flash-tiny": {
        "emb_dim": 256,
        "n_layers": 4,
        "n_heads": 8,
        "n_kv_heads": 8,
        "mlp_dim": 682,
    },
    "gemini-nano-1": {
        "emb_dim": 512,
        "n_layers": 6,
        "n_heads": 8,
        "n_kv_heads": 4,
        "mlp_dim": 1408,
    },
    "gemini-nano-2": {
        "emb_dim": 768,
        "n_layers": 8,
        "n_heads": 12,
        "n_kv_heads": 4,
        "mlp_dim": 2048,
    },
    "gemma-2b": {
        "emb_dim": 2048,
        "n_layers": 18,
        "n_heads": 8,
        "n_kv_heads": 1,
        "mlp_dim": 16384,
    },
    "gemma-7b": {
        "emb_dim": 3072,
        "n_layers": 28,
        "n_heads": 16,
        "n_kv_heads": 16,
        "mlp_dim": 24576,
    },
    "gemini-1.5-flash": {
        "emb_dim": 3584,
        "n_layers": 42,
        "n_heads": 16,
        "n_kv_heads": 8,
        "mlp_dim": 14336,
    },
}

GEMINI_1_5_FLASH_CONFIG = {**BASE_CONFIG, **GEMINI_MODEL_CONFIGS["gemini-flash-tiny"]}


class RotatoryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x):
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, seq_len):
        cos = self.cos_cached[:seq_len, :].to(dtype=x.dtype).unsqueeze(0).unsqueeze(1)
        sin = self.sin_cached[:seq_len, :].to(dtype=x.dtype).unsqueeze(0).unsqueeze(1)
        return (x * cos) + (self._rotate_half(x) * sin)

class GQA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.n_kv_heads = cfg["n_kv_heads"]
        self.head_dim = self.emb_dim // self.n_heads

        self.num_queries_per_kv = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(self.emb_dim, self.n_heads * self.head_dim, bias=cfg["qkv_bias"])
        self.k_proj = nn.Linear(self.emb_dim, self.n_kv_heads * self.head_dim, bias=cfg["qkv_bias"])
        self.v_proj = nn.Linear(self.emb_dim, self.n_kv_heads * self.head_dim, bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.emb_dim, bias=False)

        self.rope = RotatoryEmbedding(dim=self.head_dim, max_seq_len=cfg["context_length"])

    def forward(self, x):
        b, s, c = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, seq_len=s)
        k = self.rope(k, seq_len=s)

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        mask = torch.triu(torch.full((s, s), float('-inf'), device=x.device, dtype=x.dtype), diagonal=1)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_weights = attn_weights.to(x.dtype) + mask.unsqueeze(0).unsqueeze(1)
        attn_weights = F.softmax(attn_weights, dim=-1)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(b, s, -1)

        return self.out_proj(context)

class GeminiSwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_gate = nn.Linear(cfg["emb_dim"], cfg["mlp_dim"], bias=False)
        self.w_up   = nn.Linear(cfg["emb_dim"], cfg["mlp_dim"], bias=False)
        self.w_down = nn.Linear(cfg["mlp_dim"], cfg["emb_dim"], bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class DummyRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed * self.weight

class DummyGeminiFlashBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = DummyRMSNorm(cfg["emb_dim"])
        self.ffn_norm  = DummyRMSNorm(cfg["emb_dim"])
        self.attn = GQA(cfg)
        self.ffn  = GeminiSwiGLU(cfg)

    def forward(self, x):
        residual_attn = self.attn(self.attn_norm(x))
        residual_ffn  = self.ffn(self.ffn_norm(x))
        return x + residual_attn + residual_ffn

class DummyGemini15FlashModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyGeminiFlashBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyRMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        x = self.tok_emb(in_idx)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
