from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    vision_hidden_size: int
    text_hidden_size: int
    num_image_tokens: int
    image_token_id: int


class VisionToTextAdapter(nn.Module):
    """Maps vision encoder hidden states to LLM embedding space."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.num_image_tokens = num_image_tokens

        self.network = nn.Sequential(
            nn.LayerNorm(vision_hidden_size),
            nn.Linear(vision_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size),
        )

    def forward(self, vision_hidden_states: torch.Tensor) -> torch.Tensor:
        """Return visual embeddings [B, num_image_tokens, text_hidden_size]."""
        if vision_hidden_states.size(1) != self.num_image_tokens:
            x = vision_hidden_states.permute(0, 2, 1)  # [B, D, L_old]
            x = F.interpolate(x, size=self.num_image_tokens, mode="linear", align_corners=False)
            vision_hidden_states = x.permute(0, 2, 1)  # [B, num_image_tokens, D]
            
        return self.network(vision_hidden_states)


def merge_visual_embeddings(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Replace embeddings at <image> token positions with visual embeddings."""
    image_mask = input_ids == image_token_id
    
    total_expected = visual_embeds.size(0) * visual_embeds.size(1)
    if image_mask.sum().item() != total_expected:
        raise ValueError("Mismatch between detected image tokens and provided visual embeddings shape.")

    output_embeds = input_embeds.clone()
    output_embeds[image_mask] = visual_embeds.view(-1, visual_embeds.size(-1))
    
    return output_embeds


class MathVLM(nn.Module):
    """Thin wrapper around vision encoder, adapter and language model."""

    def __init__(self, vision_encoder: nn.Module, language_model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        self.adapter = VisionToTextAdapter(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_image_tokens=config.num_image_tokens,
        )

    def freeze_backbones(self) -> None:
        """Freeze vision encoder and language model parameters."""
        for model in (self.vision_encoder, self.language_model):
            for param in model.parameters():
                param.requires_grad = False

    def _prepare_multimodal_embeddings(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Helper to extract, project and merge text and visual features."""
        input_ids = batch["input_ids"]
        
        raw_features = self._extract_visual_features(batch["pixel_values"])
        visual_embeds = self.adapter(raw_features)
        
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        return merge_visual_embeddings(
            input_embeds=text_embeds,
            input_ids=input_ids,
            visual_embeds=visual_embeds,
            image_token_id=self.config.image_token_id,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        """Forward pass with loss."""
        inputs_embeds = self._prepare_multimodal_embeddings(batch)
        
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )

    @torch.no_grad()
    def generate(self, batch: dict[str, torch.Tensor], **generation_kwargs: Any) -> torch.Tensor:
        """Generate answer token ids."""
        inputs_embeds = self._prepare_multimodal_embeddings(batch)
        
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            **generation_kwargs,
        )

    def _extract_visual_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Processes pixel inputs (handling optional tiling) and returns encoder states."""
        if pixel_values.ndim == 5:
            b, t, c, h, w = pixel_values.shape
            flat_pixels = pixel_values.view(b * t, c, h, w)
            
            outputs = self.vision_encoder(pixel_values=flat_pixels)
            hidden = getattr(outputs, "last_hidden_state", outputs[0])
            
            hidden = hidden.view(b, t * hidden.size(1), hidden.size(2))
        else:
            outputs = self.vision_encoder(pixel_values=pixel_values)
            hidden = getattr(outputs, "last_hidden_state", outputs[0])
            
        return hidden