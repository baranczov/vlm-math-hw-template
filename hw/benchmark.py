from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml

from hw.constants import CHOICES
from hw.dataset import MathVQADataset
from hw.train import _move_batch, _resolve_device, build_components


def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract multiple-choice answer letter from model output."""
    cleaned = normalize_text(text)
    choice_chars = "".join(choices).lower()
    
    trigger_pattern = rf"(?:answer|ответ|correct|правильный|выбор)[:\s\-]*\(?([{choice_chars}])\)??"
    match = re.search(trigger_pattern, cleaned)
    if match:
        return match.group(1).upper()
        
    fallback_pattern = rf"\b\(?([{choice_chars}])\)?\b"
    match = re.search(fallback_pattern, cleaned)
    if match:
        return match.group(1).upper()
        
    return None


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop."""
    data_config = config.get("data", {})
    inference_config = config.get("inference", {})
    
    manifest_path = data_config.get("eval_manifest")
    split = data_config.get("split", "dev")
    if toy:
        manifest_path = "assets/toy_math_vqa/manifest.jsonl"
        split = "dev"

    dataset = MathVQADataset(
        manifest_path=manifest_path,
        split=str(split),
        max_samples=data_config.get("max_samples"),
    )
    
    tokenizer, model, processor = build_components(config)
    device = _resolve_device(str(inference_config.get("device", "cpu")))
    
    model.to(device)
    model.eval()

    results = []
    max_tokens = int(inference_config.get("max_new_tokens", 16))
    do_sample = bool(inference_config.get("do_sample", False))

    for sample in dataset:
        gold_answer = sample.answer
        object.__setattr__(sample, "answer", "") 
        
        features = processor(sample)
        batch = processor.collate([features])
        batch = _move_batch(batch, device)
        
        with torch.no_grad():
            output_ids = model.generate(
                batch,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
            )
            
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        pred_letter = parse_mc_answer(generated_text)
        
        results.append({
            "id": sample.id,
            "question": sample.question,
            "prompt": build_benchmark_prompt(sample.question, sample.options),
            "prediction": pred_letter,
            "raw_output": generated_text,
            "answer": gold_answer,
            "subject": sample.subject,
        })

    output_path = inference_config.get("output_path")
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return compute_accuracy(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()