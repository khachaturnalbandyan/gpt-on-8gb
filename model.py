"""GPT for the 8 GB scaling runs. Same design as V1, plus gradient checkpointing."""
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    gradient_checkpointing: bool = False

    def to_dict(self):
        return asdict(self)


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.query_key_value = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.projection = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, length, width = x.shape
        query, key, value = self.query_key_value(x).chunk(3, dim=-1)
        query, key, value = (
            part.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
            for part in (query, key, value)
        )
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        merged = attended.transpose(1, 2).reshape(batch, length, width)
        return self.residual_dropout(self.projection(merged))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.n_embd)
        self.attention = SelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.n_embd)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None):
        length = tokens.shape[1]
        if length > self.config.block_size:
            raise ValueError(f"Sequence length {length} exceeds block size {self.config.block_size}")
        positions = torch.arange(length, device=tokens.device)
        x = self.embedding_dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        logits = self.head(self.final_norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            logits, _ = self(tokens[:, -self.config.block_size :])
            logits = logits[:, -1, :] / temperature
            threshold = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, [-1]]
            filtered = logits.masked_fill(logits < threshold, float("-inf"))
            next_token = torch.multinomial(F.softmax(filtered, dim=-1), num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
