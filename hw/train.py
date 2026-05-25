from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hw.constants import IMAGE_TOKEN
from hw.dataset import MathVQADataset
from hw.model import MathVLM, ModelConfig
from hw.processor import MathVLMProcessor, ProcessorConfig


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ProcessingDatasetWrapper(Dataset):
    """Обертка над датасетом для параллельного процессинга через num_workers."""
    def __init__(self, base_dataset: MathVQADataset, processor: MathVLMProcessor):
        self.base_dataset = base_dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.base_dataset[idx]
        return self.processor(sample)


def train_one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Run one optimization step and return scalar loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    outputs = model(batch)
    loss = outputs["loss"]
    
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Loss exploded or became NaN: {loss.item()}")
        
    loss.backward()
    optimizer.step()
    
    return loss.item()


def run_training(config: dict[str, Any], fast_train: bool = False) -> None:
    """Main training entry point."""
    data_config = config.get("data", {})
    trainer_config = config.get("trainer", {})

    device = _resolve_device(str(trainer_config.get("device", "cpu")))
    dtype = _resolve_dtype(str(trainer_config.get("dtype", "float32")), device)

    tokenizer, model, processor = build_components(config)
    
    raw_dataset = MathVQADataset(
        manifest_path=data_config["train_manifest"],
        split=str(data_config.get("split", "train")),
        max_samples=data_config.get("max_samples"),
    )
    processed_dataset = ProcessingDatasetWrapper(raw_dataset, processor)
    
    loader = DataLoader(
        processed_dataset,
        batch_size=int(trainer_cfg.get("local_batch_size", 1)) if 'trainer_cfg' in locals() else int(trainer_config.get("local_batch_size", 1)),
        shuffle=True,
        num_workers=int(trainer_config.get("num_workers", 0)),
        collate_fn=processor.collate,
    )

    model.to(device=device, dtype=dtype)
    
    if config.get("model", {}).get("freeze_vision", True) or config.get("model", {}).get("freeze_llm", True):
        model.freeze_backbones()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(trainer_config.get("learning_rate", 5e-4)),
        weight_decay=float(trainer_config.get("weight_decay", 0.0)),
    )

    max_steps = 1 if fast_train else int(trainer_config.get("max_steps", 100))
    epochs = int(trainer_config.get("num_train_epochs", 1))
    
    global_step = 0
    keep_training = True
    
    for epoch in range(epochs):
        if not keep_training:
            break
        for batch in loader:
            batch = _move_batch(batch, device)
            train_one_step(model, batch, optimizer)
            
            global_step += 1
            if global_step >= max_steps:
                keep_training = False
                break

    save_path = trainer_config.get("save_checkpoint_path")
    if save_path:
        checkpoint_p = Path(save_path)
        checkpoint_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.adapter.state_dict(), checkpoint_p)


class SimpleTokenizer:
    def __init__(self) -> None:
        self.encoder_map = {"<pad>": 0, "<eos>": 1, IMAGE_TOKEN: 2}
        self.decoder_map = {0: "<pad>", 1: "<eos>", 2: IMAGE_TOKEN}
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = text.replace("\n", " ").split()
        ids = []
        for token in tokens:
            if token not in self.encoder_map:
                new_id = len(self.encoder_map)
                self.encoder_map[token] = new_id
                self.decoder_map[new_id] = token
            ids.append(self.encoder_map[token])
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def __call__(self, text: str, add_special_tokens: bool = False, truncation: bool = False, max_length: int | None = None) -> dict[str, list[int]]:
        token_ids = self.encode(text, add_special_tokens=add_special_tokens)
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return {"input_ids": token_ids, "attention_mask": [1] * len(token_ids)}

    def decode(self, ids: torch.Tensor | list[int], skip_special_tokens: bool = True) -> str:
        id_list = ids.tolist() if isinstance(ids, torch.Tensor) else ids
        decoded_tokens = []
        for idx in id_list:
            token_str = self.decoder_map.get(int(idx), "")
            if skip_special_tokens and token_str in {"<pad>", "<eos>", IMAGE_TOKEN}:
                continue
            decoded_tokens.append(token_str)
        return " ".join(decoded_tokens)


class SimpleVisionEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_tokens: int) -> None:
        super().__init__()
        self.spatial_pool = nn.AdaptiveAvgPool2d((num_tokens, 1))
        self.feature_proj = nn.Linear(3, hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> Any:
        x = self.spatial_pool(pixel_values).squeeze(-1)  # [B, 3, num_tokens]
        features = x.transpose(1, 2)  # [B, num_tokens, 3]
        return SimpleVisionOutput(self.feature_proj(features))


class SimpleVisionOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embeddings

    def forward(self, input_ids: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)
            
        logits = self.output_head(inputs_embeds)
        output_dict = {"logits": logits}
        
        if labels is not None:
            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.view(-1)
            output_dict["loss"] = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
            
        return output_dict

    def generate(self, input_ids: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None, max_new_tokens: int = 8, **kwargs: Any) -> torch.Tensor:
        if inputs_embeds is not None:
            seq_buffer = inputs_embeds[:, -1:, :]
            b_size = inputs_embeds.size(0)
            dev = inputs_embeds.device
        else:
            seq_buffer = self.token_embeddings(input_ids[:, -1:])
            b_size = input_ids.size(0)
            dev = input_ids.device

        generated_tokens = []
        for _ in range(max_new_tokens):
            step_logits = self.output_head(seq_buffer[:, -1])
            next_token = step_logits.argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            
            next_embed = self.token_embeddings(next_token)
            seq_buffer = torch.cat([seq_buffer, next_embed], dim=1)
            
        return torch.cat(generated_tokens, dim=1) if generated_tokens else torch.empty(b_size, 0, dtype=torch.long, device=dev)


def build_components(config: dict[str, Any]) -> tuple[Any, MathVLM, MathVLMProcessor]:
    proc_config = ProcessorConfig(**config.get("processor", {}))
    tokenizer = SimpleTokenizer()
    
    v_encoder = SimpleVisionEncoder(hidden_size=32, num_tokens=proc_config.num_image_tokens)
    l_model = SimpleLanguageModel(vocab_size=4096, hidden_size=64)
    
    model_config = ModelConfig(
        vision_hidden_size=32,
        text_hidden_size=64,
        num_image_tokens=proc_config.num_image_tokens,
        image_token_id=tokenizer.encoder_map[IMAGE_TOKEN],
    )
    return tokenizer, MathVLM(v_encoder, l_model, model_config), MathVLMProcessor(tokenizer, proc_config)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
        
    clean_name = name.lower().replace("float", "fp")
    mapping = {
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return mapping.get(clean_name, torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()