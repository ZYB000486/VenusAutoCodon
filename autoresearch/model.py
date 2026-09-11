from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .vocab import (
    SRC_PAD_ID,
    SRC_VOCAB,
    TGT_BOS_ID,
    TGT_PAD_ID,
    TGT_VOCAB,
    build_constraint_bias,
)


@dataclass
class Seq2SeqConfig:
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 512
    use_codon_constraint: bool = True

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def config_from_dict(payload: dict[str, Any]) -> Seq2SeqConfig:
    defaults = Seq2SeqConfig().to_dict()
    valid_names = {field.name for field in fields(Seq2SeqConfig)}
    defaults.update({key: value for key, value in payload.items() if key in valid_names})
    return Seq2SeqConfig(**defaults)


class StandardSeq2SeqTransformer(nn.Module):
    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        self.src_embedding = nn.Embedding(len(SRC_VOCAB), config.d_model, padding_idx=SRC_PAD_ID)
        self.tgt_embedding = nn.Embedding(len(TGT_VOCAB), config.d_model, padding_idx=TGT_PAD_ID)
        self.src_position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.tgt_position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.output_projection = nn.Linear(config.d_model, len(TGT_VOCAB))

        constraint_bias = build_constraint_bias()
        self.register_buffer("constraint_bias", constraint_bias, persistent=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.src_position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.tgt_position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def _embed_source(self, src_ids: torch.Tensor) -> torch.Tensor:
        seq_len = src_ids.size(1)
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Source length {seq_len} exceeds max_seq_len={self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=src_ids.device).unsqueeze(0)
        hidden = self.src_embedding(src_ids) * (self.d_model**0.5)
        hidden = hidden + self.src_position_embedding(positions)
        return self.dropout(hidden)

    def _embed_target(self, tgt_ids: torch.Tensor, *, start_pos: int = 0) -> torch.Tensor:
        seq_len = tgt_ids.size(1)
        max_pos = start_pos + seq_len
        if max_pos > self.config.max_seq_len:
            raise ValueError(f"Target length {max_pos} exceeds max_seq_len={self.config.max_seq_len}")
        positions = torch.arange(start_pos, max_pos, device=tgt_ids.device).unsqueeze(0)
        hidden = self.tgt_embedding(tgt_ids) * (self.d_model**0.5)
        hidden = hidden + self.tgt_position_embedding(positions)
        return self.dropout(hidden)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)

    def _encode_memory(self, src_ids: torch.Tensor, src_padding_mask: torch.Tensor | None) -> torch.Tensor:
        src_hidden = self._embed_source(src_ids)
        return self.transformer.encoder(src_hidden, src_key_padding_mask=src_padding_mask)

    def _decode_full(
        self,
        tgt_hidden: torch.Tensor,
        memory: torch.Tensor,
        *,
        src_padding_mask: torch.Tensor | None,
        tgt_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        tgt_mask = self._causal_mask(tgt_hidden.size(1), tgt_hidden.device)
        return self.transformer.decoder(
            tgt_hidden,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

    def _ff_block(self, layer: nn.TransformerDecoderLayer, hidden: torch.Tensor) -> torch.Tensor:
        ff_hidden = layer.linear1(hidden)
        ff_hidden = layer.activation(ff_hidden)
        ff_hidden = layer.dropout(ff_hidden)
        ff_hidden = layer.linear2(ff_hidden)
        return layer.dropout3(ff_hidden)

    def _decode_next_token(
        self,
        current_hidden: torch.Tensor,
        memory: torch.Tensor,
        *,
        src_padding_mask: torch.Tensor | None,
        layer_caches: list[torch.Tensor | None],
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        hidden = current_hidden
        for layer_idx, layer in enumerate(self.transformer.decoder.layers):
            if not layer.norm_first:
                raise RuntimeError("Cached decoding expects norm_first=True")

            current_norm = layer.norm1(hidden)
            cached_norm = layer_caches[layer_idx]
            self_attn_kv = current_norm if cached_norm is None else torch.cat([cached_norm, current_norm], dim=1)

            self_attn_out = layer.self_attn(
                current_norm,
                self_attn_kv,
                self_attn_kv,
                need_weights=False,
            )[0]
            hidden = hidden + layer.dropout1(self_attn_out)

            cross_query = layer.norm2(hidden)
            cross_attn_out = layer.multihead_attn(
                cross_query,
                memory,
                memory,
                key_padding_mask=src_padding_mask,
                need_weights=False,
            )[0]
            hidden = hidden + layer.dropout2(cross_attn_out)

            ff_input = layer.norm3(hidden)
            hidden = hidden + self._ff_block(layer, ff_input)
            layer_caches[layer_idx] = self_attn_kv

        if self.transformer.decoder.norm is not None:
            hidden = self.transformer.decoder.norm(hidden)

        return hidden, layer_caches

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        *,
        src_padding_mask: torch.Tensor | None = None,
        tgt_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt_len = tgt_input_ids.size(1)
        if tgt_len > src_ids.size(1):
            raise ValueError("Target sequence cannot be longer than aligned source sequence")

        memory = self._encode_memory(src_ids, src_padding_mask)
        tgt_hidden = self._embed_target(tgt_input_ids)
        hidden = self._decode_full(
            tgt_hidden,
            memory,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )
        logits = self.output_projection(hidden)
        if self.config.use_codon_constraint:
            aligned_src_ids = src_ids[:, :tgt_len]
            logits = logits + self.constraint_bias[aligned_src_ids]
        return logits

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        *,
        src_padding_mask: torch.Tensor | None = None,
        sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        if src_padding_mask is None:
            src_padding_mask = src_ids.eq(SRC_PAD_ID)

        if not use_cache:
            return self._generate_without_cache(
                src_ids,
                src_padding_mask=src_padding_mask,
                sample=sample,
                temperature=temperature,
                top_k=top_k,
            )

        batch_size, max_src_len = src_ids.shape
        lengths = (~src_padding_mask).sum(dim=1)
        memory = self._encode_memory(src_ids, src_padding_mask)
        layer_caches: list[torch.Tensor | None] = [None] * len(self.transformer.decoder.layers)
        prev_tokens = torch.full((batch_size, 1), fill_value=TGT_BOS_ID, dtype=torch.long, device=src_ids.device)
        generated: list[torch.Tensor] = []

        for step in range(max_src_len):
            current_hidden = self._embed_target(prev_tokens, start_pos=step)
            hidden, layer_caches = self._decode_next_token(
                current_hidden,
                memory,
                src_padding_mask=src_padding_mask,
                layer_caches=layer_caches,
            )
            next_logits = self.output_projection(hidden).squeeze(1)
            if self.config.use_codon_constraint:
                next_logits = next_logits + self.constraint_bias[src_ids[:, step]]

            if sample:
                if temperature <= 0:
                    raise ValueError("temperature must be positive when sampling")
                next_logits = next_logits / temperature
                if top_k > 0:
                    values, indices = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)), dim=-1)
                    filtered = torch.full_like(next_logits, float("-inf"))
                    filtered.scatter_(1, indices, values)
                    next_logits = filtered
                probs = F.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = next_logits.argmax(dim=-1)

            active = step < lengths
            next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, TGT_PAD_ID))
            generated.append(next_tokens)
            prev_tokens = next_tokens.unsqueeze(1)

        return torch.stack(generated, dim=1)

    def _generate_without_cache(
        self,
        src_ids: torch.Tensor,
        *,
        src_padding_mask: torch.Tensor,
        sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        batch_size, max_src_len = src_ids.shape
        lengths = (~src_padding_mask).sum(dim=1)
        generated = torch.full((batch_size, 1), fill_value=TGT_BOS_ID, dtype=torch.long, device=src_ids.device)

        for step in range(max_src_len):
            logits = self(
                src_ids,
                generated,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=generated.eq(TGT_PAD_ID),
            )
            next_logits = logits[:, -1, :]
            if sample:
                if temperature <= 0:
                    raise ValueError("temperature must be positive when sampling")
                next_logits = next_logits / temperature
                if top_k > 0:
                    values, indices = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)), dim=-1)
                    filtered = torch.full_like(next_logits, float("-inf"))
                    filtered.scatter_(1, indices, values)
                    next_logits = filtered
                probs = F.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = next_logits.argmax(dim=-1)

            active = step < lengths
            next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, TGT_PAD_ID))
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
        return generated[:, 1:]


def build_model(config: Seq2SeqConfig) -> StandardSeq2SeqTransformer:
    return StandardSeq2SeqTransformer(config)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[StandardSeq2SeqTransformer, Seq2SeqConfig, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=map_location)
    config = config_from_dict(payload["model_config"])
    model = build_model(config)
    model.load_state_dict(payload["model_state"])
    return model, config, payload


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (param for param in params if param.requires_grad)
    return sum(param.numel() for param in params)
