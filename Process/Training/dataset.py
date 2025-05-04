# dataset.py
"""JSONL ➜ PyTorch dataset for recipe-name generation.

Ground-truth format
-------------------
<recipe> name₁ <recipe> name₂ … <eos>
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

RECIPE_TOKEN = "<recipe>"

class RecipeDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer_name: str,
        max_length: int = 512,
        live_eval: bool = False,
    ):
        """
        Recipe Dataset for training and inference.
        
        Args:
            path: Path to JSONL file with recipe data
            tokenizer_name: Name or path of the tokenizer to use
            max_length: Maximum sequence length
            live_eval: If True, optimizes for real-time inference
        """
        self.live_eval = live_eval
        
        # For live_eval mode, records can be empty as they'll be provided during inference
        if not live_eval and path:
            try:
                self.records: List[Dict] = [
                    json.loads(line) for line in Path(path).open(encoding="utf-8")
                ]
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading dataset: {e}")
                self.records = []
        else:
            self.records = []

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # add `<recipe>` if missing
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [RECIPE_TOKEN]}
        )

        if self.tokenizer.pad_token is None:          # GPT-2 quirk
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length

    @staticmethod
    def _join(items: List[str]) -> str:
        """Join items with commas and transform spaces to underscores."""
        return ", ".join(i.lower().replace(" ", "_") for i in items)

    def _build_prompt(self, rec: Dict) -> str:
        """Build input prompt from ingredients and tags."""
        ings = self._join(rec.get("ingredients", []))
        tags = self._join(rec.get("tags", []))
        return (f"Ingredients: {ings}\n"
                f"Tags: {tags}\n"
                f"Recipe name:")

    def _build_label_seq(self, rec: Dict) -> str:
        """Build output sequence with recipe token before each name."""
        # prepend space before each <recipe> for clean detokenisation
        return "".join(f" {RECIPE_TOKEN} {n}" for n in rec.get("names", []))

    def prepare_for_inference(self, ingredients: List[str], tags: List[str]) -> Dict[str, torch.Tensor]:
        """
        Prepare a single sample for live inference.
        
        Args:
            ingredients: List of ingredient names
            tags: List of tags
            
        Returns:
            Dict with tokenized inputs ready for model inference
        """
        rec = {"ingredients": ingredients, "tags": tags}
        prompt = self._build_prompt(rec)
        
        # For inference, we don't need labels - just prepare the input
        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        prompt      = self._build_prompt(rec)
        label_part  = self._build_label_seq(rec) + self.tokenizer.eos_token
        full_text   = prompt + label_part

        # Use padding=True to allow DataCollator to do dynamic padding
        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Compute where to start labeling
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

        # Flatten tensors from shape (1, max_len) to (max_len,)
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

