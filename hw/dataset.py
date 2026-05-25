from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MathVQASample:
    """One visual-math QA example.

    Attributes:
        id: Stable example id.
        image: PIL image in RGB mode.
        question: User question without hidden answer.
        options: Multiple-choice options, e.g. ["A) ...", "B) ..."].
        answer: Gold answer, usually "A"/"B"/"C"/"D" for public toy data.
        subject: Topic label, e.g. geometry/algebra/plots.
        source: Dataset/source label.
    """

    id: str
    image: Image.Image
    question: str
    options: list[str]
    answer: str
    subject: str
    source: str = "unknown"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a jsonl file."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}") from exc
    return rows


def sanitize_question(text: str) -> str:
    """Remove image/control tokens that must not appear in raw questions."""
    for token in ("<image>", "<image_start>", "<image_end>"):
        text = text.replace(token, "")
    return " ".join(text.split())


class MathVQADataset(Dataset[MathVQASample]):
    """Dataset for manifest-based visual mathematical QA.

    Expected manifest fields:
        id, split, image, question, options, answer, subject, source(optional)
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        max_samples: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.split = split
        self.max_samples = max_samples

        raw_rows = load_jsonl(self.manifest_path)
        
        self._data = [row for row in raw_rows if row.get("split") == self.split]
        
        if self.max_samples is not None:
            self._data = self._data[: self.max_samples]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> MathVQASample:
        item = self._data[idx]
        
        image_path = self.root / item["image"]
        img = Image.open(image_path).convert("RGB")
        
        clean_question = sanitize_question(str(item.get("question", "")))
        
        options_list = [str(opt) for opt in item.get("options", [])]

        return MathVQASample(
            id=str(item.get("id", idx)),
            image=img,
            question=clean_question,
            options=options_list,
            answer=str(item.get("answer", "")),
            subject=str(item.get("subject", "unknown")),
            source=str(item.get("source", "unknown")),
        )