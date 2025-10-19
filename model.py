import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

import math

torch.manual_seed(1024)


@dataclass
class GPTConfig:
    block_size: int = 512
    batch_size: int = 12
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    vocab_size: int = 50274
    head_size: Optional[int] = None

    def __post_init__(self):
        if self.head_size is None:
            if self.n_embd % self.n_head != 0:
                raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
            self.head_size = self.n_embd // self.n_head
        else:
            if self.head_size * self.n_head != self.n_embd:
                raise ValueError(f"head_size * n_head ({self.head_size * self.n_head}) != n_embd ({self.n_embd})")


class MultiHeadAttentionEfficient(nn.Module):
    """
    Efficient multi-head self-attention: qkv projection, reshape to heads, compute attention,
    merge heads and out projection.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.head_size
        self.n_embd = config.n_embd
        assert self.n_head * self.head_size == self.n_embd, "Expect n_head * head_size == n_embd"

        # project once to q,k,v (each of size n_embd)
        self.qkv_proj = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "attention_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).bool()
        )

    def forward(self, x):
        # x: (B, T, C) where C == n_embd
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3*n_embd)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, n_embd)

        # reshape to (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # scaled dot-product attention
        # scores: (B, n_head, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)

        # apply causal mask: mask shape (T, T) -> (1, 1, T, T) to broadcast to scores
        mask = self.attention_mask[:T, :T]  # (T, T)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-1e9'))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # attn @ v -> (B, n_head, T, head_size)
        out = torch.matmul(attn, v)

        # merge heads -> (B, T, n_embd)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_embd)

        out = self.out_proj(out)
        out = self.out_dropout(out)
        return out


class FeedForward(nn.Module):
    """
    Feed-Forward (MLP) block: Linear -> GELU -> Linear
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        n_embd = config.n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block with Pre-LayerNorm
    x -> x + att(LN(x)); x -> x + ffn(LN(x))
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.att = MultiHeadAttentionEfficient(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    GPT-2 model
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # token + position embeddings
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        # stack of transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # final norm and language model head
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init weights and tie token embedding with lm_head
        self.apply(self._init_weights)
        # weight tying: make output projection share same weight matrix as token embedding
        self.lm_head.weight = self.token_embedding_table.weight

    def _init_weights(self, module):
        # follow common practice: normal_(std=0.02)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) LongTensor
        targets: (B, T) LongTensor or None
        returns: logits (B, T, V), loss (scalar) or None
        """
        B, T = idx.size()
        device = idx.device

        # token & position embeddings
        token_emb = self.token_embedding_table(idx)         # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = token_emb + pos_emb  # broadcast to (B, T, n_embd)

        # transformer blocks (explicit loop for clarity)
        for block in self.blocks:
            x = block(x)  # each block preserves shape (B, T, n_embd)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # flatten for cross_entropy
            logits_flat = logits.view(B * T, -1)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, do_sample=True):
        """
        Simple generation loop. For learning purposes only.
        - idx: (B, T0) initial context (LongTensor)
        - returns: (B, T0 + max_new_tokens)
        """
        self.eval()
        device = next(self.parameters()).device
        idx = idx.to(device)

        for _ in range(max_new_tokens):
            # crop context to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)            # (B, T, V)
            logits = logits[:, -1, :]             # (B, V)
            logits = logits / max(temperature, 1e-8)

            # top-k filtering (optional)
            if top_k is not None and top_k > 0:
                vals, _ = torch.topk(logits, top_k, dim=-1)
                min_vals = vals[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_vals, torch.full_like(logits, -1e9), logits)

            probs = F.softmax(logits, dim=-1)

            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)  # (B,1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            idx = torch.cat([idx, next_token], dim=1)

        self.train()
        return idx


class MyDataset(Dataset):
    """
    Simple dataset wrapper using tiktoken .
    For real training, use streaming / HF datasets to scale.
    """
    def __init__(self, path, block_size=512, max_lines=1000):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.max_lines = max_lines

        # encode corpus (simple, small-scale)
        self.encoded_data = []
        self.eos_token = self.enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        raw_texts = []
        import json
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    text = json.loads(line.strip())["text"]
                    raw_texts.append(text)
                except Exception:
                    continue

        # concatenate and chunk into block_size+1 windows (x,y)
        full_encoded = []
        for text in raw_texts:
            enc_text = self.enc.encode(text)
            full_encoded.extend(enc_text + [self.eos_token])

        # create fixed-length chunks (each chunk yields x=chunk[:-1], y=chunk[1:])
        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i:i + self.block_size + 1]
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, idx):
        return self.enc.decode(idx)


# ----------------------------
# sanity check
# ----------------------------
if __name__ == "__main__":
    cfg = GPTConfig(block_size=64, batch_size=2, n_layer=2, n_head=4, n_embd=256)
    model = GPT(cfg)
    B, T = 2, 16
    x = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(x, targets=x)
    print("logits.shape:", logits.shape)  # (B, T, V)
    print("loss:", loss.item())
    # generate example (short)
    start = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(start, max_new_tokens=8, temperature=1.0, top_k=50)
    print("generated shape:", out.shape)
