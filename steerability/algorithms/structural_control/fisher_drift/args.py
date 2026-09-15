"""Arguments for prior-Fisher drift-budget steering."""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class FisherDriftArgs(BaseArgs):
    """Arguments for `FisherDrift`.

    Both datasets must be finite and indexable. Rows contain one-dimensional `input_ids`
    and may contain matching `attention_mask` and `labels` values. Missing masks default to
    ones and missing labels default to `input_ids`. A custom `data_collator` may turn rows
    into the same three-tensor mapping.

    Attributes:
        prior_dataset: Reference examples defining behavior to preserve.
        train_dataset: Supervised examples defining the target update.
        kappa: Predicted quadratic drift budget.
        damping: Positive diagonal added to the Fisher estimate.
        fisher_num_samples: Maximum number of reference rows sampled for Fisher contexts.
        fisher_seed: Seed for reference-row, context-position, and pseudo-label sampling.
        per_device_batch_size: Batch size used to accumulate the target gradient.
        max_length: Maximum number of tokens retained from each example.
        data_collator: Optional callable collating a list of dataset rows.
        r: LoRA rank used when attaching a new adapter.
        lora_alpha: LoRA scaling parameter.
        target_modules: Module-name suffixes targeted by a new LoRA adapter.
        lora_dropout: LoRA dropout probability. Fisher and target gradients run in eval mode.
        adapter_name: Adapter to reuse on a PEFT model, or name for a newly attached adapter.
        output_dir: Directory receiving the final PEFT adapter and tokenizer.
    """

    prior_dataset: Any = None
    train_dataset: Any = None
    kappa: float = 1e-4
    damping: float = 1e-5
    fisher_num_samples: int = 128
    fisher_seed: int = 42
    per_device_batch_size: int = 1
    max_length: int = 1024
    data_collator: Any | None = None

    r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.0
    adapter_name: str | None = None
    output_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if self.prior_dataset is None:
            raise ValueError("prior_dataset is required.")
        if self.train_dataset is None:
            raise ValueError("train_dataset is required.")
        for name, dataset in (("prior_dataset", self.prior_dataset), ("train_dataset", self.train_dataset)):
            if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
                raise TypeError(f"{name} must be finite and indexable.")
            if len(dataset) == 0:
                raise ValueError(f"{name} must not be empty.")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")
        if self.damping <= 0:
            raise ValueError("damping must be positive.")
        if self.fisher_num_samples <= 0:
            raise ValueError("fisher_num_samples must be positive.")
        if self.per_device_batch_size <= 0:
            raise ValueError("per_device_batch_size must be positive.")
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2.")
        if self.r <= 0:
            raise ValueError("r must be positive.")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive.")
        if not self.target_modules or not all(isinstance(name, str) and name for name in self.target_modules):
            raise ValueError("target_modules must contain at least one non-empty string.")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1).")
        if self.adapter_name is not None and not self.adapter_name:
            raise ValueError("adapter_name must be non-empty when provided.")

        self.output_dir = str(self.output_dir or tempfile.mkdtemp(prefix="steerability-fisher-drift-"))
