from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size]."""
        tgt_size = self.config.image_size
        img_rgb = image.convert("RGB")
        
        if self.config.num_tiles <= 1:
            resized = img_rgb.resize((tgt_size, tgt_size), Image.Resampling.BICUBIC)
            tensor = torch.from_numpy(torch.get_default_dtype() if hasattr(torch, "get_default_dtype") else None or Any)
            img_tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
            return img_tensor.unsqueeze(0)

        grid_dim = int(self.config.num_tiles ** 0.5)
        if grid_dim * grid_dim < self.config.num_tiles:
            grid_dim += 1
            
        w, h = img_rgb.size
        tile_w, tile_h = w // grid_dim, h // grid_dim
        
        tiles = []
        for i in range(grid_dim):
            for j in range(grid_dim):
                if len(tiles) >= self.config.num_tiles:
                    break
                left = j * tile_w
                upper = i * tile_h
                box = (left, upper, left + tile_w, upper + tile_h)
                
                tile = img_rgb.crop(box).resize((tgt_size, tgt_size), Image.Resampling.BICUBIC)
                tile_tensor = torch.tensor(tile, dtype=torch.float32).permute(2, 0, 1) / 255.0
                tiles.append(tile_tensor)
                
        return torch.stack(tiles, dim=0)

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options."""
        visual_sequence = " ".join([IMAGE_TOKEN] * self.config.num_image_tokens)
        options_block = "\n".join(sample.options)
        
        prompt = (
            f"{IMAGE_START_TOKEN} {visual_sequence} {IMAGE_END_TOKEN}\n"
            f"Реши визуально-математическую задачу. Выбери один вариант ответа.\n\n"
            f"Вопрос: {sample.question}\n"
            f"Варианты:\n{options_block}\n"
            f"Ответ:"
        )
        
        if include_answer:
            prompt += f" {sample.answer}"
        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample."""
        full_text = self.build_prompt(sample, include_answer=True)
        prompt_text = self.build_prompt(sample, include_answer=False)
        
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        
        if len(full_ids) > self.config.max_length:
            full_ids = full_ids[:self.config.max_length]
            prompt_ids = prompt_ids[:min(len(prompt_ids), self.config.max_length)]
            
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        labels = torch.full_like(input_ids, self.config.ignore_index)
        answer_start_idx = len(prompt_ids)
        
        if answer_start_idx < len(input_ids):
            labels[answer_start_idx:] = input_ids[answer_start_idx:]
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values."""
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        if pad_id is None:
            pad_id = 0
            
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch], 
            batch_first=True, 
            padding_value=pad_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch], 
            batch_first=True, 
            padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch], 
            batch_first=True, 
            padding_value=self.config.ignore_index
        )
        
        pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }