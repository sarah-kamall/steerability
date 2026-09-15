"""Prior-Fisher drift-budget structural control."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.execution.contracts import Capability
from steerability.algorithms.core.execution.payloads import LoRAArtifact
from steerability.algorithms.structural_control.base import StructuralControl
from steerability.utils.tokenization import ensure_pad_token

from .args import FisherDriftArgs
from .estimation import compute_target_gradient, estimate_diagonal_fisher
from .update import drift_budget_update


class FisherDrift(StructuralControl):
    """Apply a prior-Fisher-preconditioned LoRA update under a drift budget.

    The incoming model is the reference parameter point `theta_A`. `prior_dataset` estimates
    a true diagonal Fisher over sampled next-token contexts, while `train_dataset` supplies
    the supervised target gradient. The control applies one update whose damped quadratic
    drift is `kappa`, saves the resulting PEFT adapter, and returns the adapted model.

    The quadratic budget is a local approximation; it does not guarantee that measured KL
    divergence on the reference distribution is at most `kappa`.

    Reference:

        - Local prior-Fisher experiment, `prior_fisher/methods.py`, `drift_budget_step`.
    """

    Args = FisherDriftArgs
    supports_batching = True

    def _configure(self) -> None:
        self._artifact: LoRAArtifact | None = None
        self._resolved_adapter_name: str | None = None
        self.diagnostics: dict[str, float | int] = {}

    def artifact_capability(self) -> Capability:
        """The final update is saved as a serveable LoRA adapter."""
        return Capability.SERVE_LORA

    def export_artifact(self) -> LoRAArtifact | None:
        """Return the saved LoRA adapter after `steer()`."""
        return self._artifact

    def export_state(self) -> dict[str, Any]:
        """Return the final adapter for `.spipe` freezing."""
        return {"artifact": self._artifact} if self._artifact is not None else {}

    def frozen_form(self, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Resolve a fitted control to the standard `load_lora` control."""
        artifact = state["artifact"]
        return "structural_control/load_lora", {
            "path": artifact,
            "base_model": artifact.base_model,
            "merge": False,
        }

    def fit_identity(self) -> dict[str, Any]:
        """Return fit-relevant configuration, excluding the output location."""
        return {
            field.name: getattr(self.args, field.name)
            for field in fields(self.args)
            if field.init and field.name != "output_dir"
        }

    def _prepare_adapter(self, model: PreTrainedModel) -> tuple[PeftModel, list[torch.nn.Parameter], str]:
        if isinstance(model, PeftModel):
            adapter_name = self.adapter_name or model.active_adapter
            if isinstance(adapter_name, (list, tuple)):
                if len(adapter_name) != 1:
                    raise ValueError("FisherDrift requires exactly one active PEFT adapter.")
                adapter_name = adapter_name[0]
            if adapter_name not in model.peft_config:
                raise ValueError(f"PEFT adapter {adapter_name!r} does not exist on the incoming model.")
            model.set_adapter(adapter_name)
        else:
            adapter_name = self.adapter_name or "fisher_drift"
            config = LoraConfig(
                r=self.r,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, config, adapter_name=adapter_name)

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        marker = f".{adapter_name}."
        selected = [
            parameter
            for name, parameter in model.named_parameters()
            if "lora_" in name and marker in name
        ]
        for parameter in selected:
            parameter.requires_grad_(True)
        if not selected:
            raise ValueError(f"No LoRA parameters were found for adapter {adapter_name!r}.")
        return model, selected, adapter_name

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs,
    ) -> PreTrainedModel:
        """Estimate the prior Fisher, apply one target update, and save the adapter.

        Args:
            model: Incoming reference model at `theta_A`.
            tokenizer: Tokenizer used by the pretokenized datasets and for padding.

        Returns:
            The PEFT model carrying the updated adapter.

        Raises:
            ValueError: If the model, tokenizer, datasets, or selected adapter are invalid.
        """
        if model is None:
            raise ValueError("FisherDrift requires an incoming model.")
        if tokenizer is None:
            raise ValueError("FisherDrift requires a tokenizer.")
        tokenizer = ensure_pad_token(tokenizer)

        base_model = getattr(model, "name_or_path", None) or getattr(
            getattr(model, "config", None), "_name_or_path", None
        )
        model, parameters, adapter_name = self._prepare_adapter(model)
        if not base_model:
            config = model.peft_config[adapter_name]
            base_model = getattr(config, "base_model_name_or_path", None)
        if not base_model:
            raise ValueError("Could not resolve the base model reference required for the LoRA artifact.")

        fisher, contexts = estimate_diagonal_fisher(
            model,
            parameters,
            self.prior_dataset,
            tokenizer,
            num_samples=self.fisher_num_samples,
            seed=self.fisher_seed,
            max_length=self.max_length,
            data_collator=self.data_collator,
        )
        gradients, target_tokens = compute_target_gradient(
            model,
            parameters,
            self.train_dataset,
            tokenizer,
            batch_size=self.per_device_batch_size,
            max_length=self.max_length,
            data_collator=self.data_collator,
        )
        updates, diagnostics = drift_budget_update(
            gradients, fisher, kappa=self.kappa, damping=self.damping,
        )
        with torch.no_grad():
            for parameter, update in zip(parameters, updates):
                parameter.add_(update.to(device=parameter.device, dtype=parameter.dtype))

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir, selected_adapters=[adapter_name])
        tokenizer.save_pretrained(output_dir)

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        self._resolved_adapter_name = adapter_name
        adapter_dir = output_dir if adapter_name == "default" else output_dir / adapter_name
        self._artifact = LoRAArtifact(path=str(adapter_dir), base_model=str(base_model))
        fisher_count = sum(diagonal.numel() for diagonal in fisher)
        fisher_sum = sum(float(diagonal.detach().float().sum().cpu()) for diagonal in fisher)
        fisher_max = max(float(diagonal.detach().float().max().cpu()) for diagonal in fisher)
        self.diagnostics = {
            **diagnostics,
            "fisher_mean": fisher_sum / fisher_count,
            "fisher_max": fisher_max,
            "contexts_sampled": contexts,
            "target_tokens": target_tokens,
        }
        return model
