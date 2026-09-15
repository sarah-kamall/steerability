"""Tests for the prior-Fisher drift-budget structural control."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from steerability.algorithms.core.execution.contracts import Capability
from steerability.algorithms.core.execution.payloads import LoRAArtifact
from steerability.algorithms.core.registry import REGISTRY
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.structural_control.fisher_drift import FisherDrift
from steerability.algorithms.structural_control.fisher_drift.estimation import (
    collate_rows,
    compute_target_gradient,
    estimate_diagonal_fisher,
)
from steerability.algorithms.structural_control.fisher_drift.update import drift_budget_update


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "</s>"

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


def _rows():
    return [
        {"input_ids": [1, 4, 5, 6], "attention_mask": [1, 1, 1, 1]},
        {"input_ids": [1, 7, 8], "attention_mask": [1, 1, 1]},
    ]


def test_registry_discovery():
    assert "fisher_drift" in REGISTRY["structural_control"]


def test_drift_budget_update_hits_quadratic_budget():
    gradients = [torch.tensor([2.0, -1.0]), torch.tensor([[0.5]])]
    fisher = [torch.tensor([4.0, 1.0]), torch.tensor([[2.0]])]

    updates, diagnostics = drift_budget_update(
        gradients, fisher, kappa=0.03, damping=0.1,
    )

    predicted = 0.5 * sum(
        ((diagonal + 0.1) * update.square()).sum()
        for diagonal, update in zip(fisher, updates)
    )
    assert float(predicted) == pytest.approx(0.03, rel=1e-5)
    assert diagnostics["predicted_drift"] == pytest.approx(0.03, rel=1e-5)
    assert sum((gradient * update).sum() for gradient, update in zip(gradients, updates)) < 0


def test_drift_budget_zero_gradient_is_noop():
    updates, diagnostics = drift_budget_update(
        [torch.zeros(3)], [torch.ones(3)], kappa=0.1, damping=1e-3,
    )
    assert torch.equal(updates[0], torch.zeros(3))
    assert diagnostics["predicted_drift"] == 0.0


def test_default_collator_masks_padding_and_preserves_label_masks():
    rows = _rows()
    rows[0]["labels"] = [-100, -100, 5, 6]
    batch = collate_rows(rows, _Tokenizer(), max_length=8)

    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert batch["labels"][0].tolist() == [-100, -100, 5, 6]
    assert batch["labels"][1].tolist() == [1, 7, 8, -100]


class _SometimesNonfiniteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 3)
        self.logit_bias = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, use_cache):
        del attention_mask, use_cache
        logits = self.logit_bias.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
        if int(input_ids[0, 0]) == 9:
            logits = logits.clone()
            logits[:, 0] = torch.full_like(logits[:, 0], float("nan"))
        return SimpleNamespace(logits=logits)


def test_fisher_skips_an_isolated_nonfinite_context():
    model = _SometimesNonfiniteModel()
    rows = [
        {"input_ids": [9, 2], "labels": [9, 2]},
        {"input_ids": [1, 2], "labels": [1, 2]},
    ]

    with pytest.warns(RuntimeWarning, match="non-finite logits"):
        fisher, contexts = estimate_diagonal_fisher(
            model,
            [model.logit_bias],
            rows,
            _Tokenizer(),
            num_samples=2,
            seed=0,
            max_length=8,
        )

    assert contexts == 1
    assert torch.isfinite(fisher[0]).all()


def test_target_gradient_ignores_nonfinite_unsupervised_positions():
    model = _SometimesNonfiniteModel()
    row = {
        "input_ids": [9, 1, 2],
        "attention_mask": [1, 1, 1],
        "labels": [-100, -100, 2],
    }

    gradients, tokens = compute_target_gradient(
        model,
        [model.logit_bias],
        [row],
        _Tokenizer(),
        batch_size=1,
        max_length=8,
    )

    assert tokens == 1
    assert torch.isfinite(gradients[0]).all()


def test_args_validation():
    with pytest.raises(ValueError, match="prior_dataset"):
        FisherDrift(train_dataset=_rows())
    with pytest.raises(ValueError, match="kappa"):
        FisherDrift(prior_dataset=_rows(), train_dataset=_rows(), kappa=0)
    with pytest.raises(ValueError, match="damping"):
        FisherDrift(prior_dataset=_rows(), train_dataset=_rows(), damping=0)


def test_tiny_llama_updates_and_exports_lora(tmp_path):
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._name_or_path = "local/tiny-llama"
    model = LlamaForCausalLM(config)
    control = FisherDrift(
        prior_dataset=_rows(),
        train_dataset=_rows(),
        fisher_num_samples=2,
        max_length=8,
        r=2,
        lora_alpha=4,
        output_dir=tmp_path / "adapter",
    )

    adapted = control.steer(model, _Tokenizer())

    assert adapted is not model
    assert control.artifact_capability() is Capability.SERVE_LORA
    artifact = control.export_artifact()
    assert isinstance(artifact, LoRAArtifact)
    assert artifact.base_model == "local/tiny-llama"
    assert (Path(artifact.path) / "adapter_config.json").exists()
    assert control.diagnostics["contexts_sampled"] == 2
    assert control.diagnostics["target_tokens"] == 5
    assert control.diagnostics["predicted_drift"] == pytest.approx(control.kappa, rel=1e-4)
    assert not any(parameter.requires_grad for parameter in adapted.parameters())
    method, kwargs = control.frozen_form(control.export_state())
    assert method == "structural_control/load_lora"
    assert kwargs["base_model"] == "local/tiny-llama"


def test_pipeline_freezes_to_load_lora(tmp_path):
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._name_or_path = "local/tiny-llama"
    control = FisherDrift(
        prior_dataset=_rows(),
        train_dataset=_rows(),
        fisher_num_samples=1,
        max_length=8,
        r=2,
        lora_alpha=4,
        output_dir=tmp_path / "adapter",
    )
    pipeline = SteeringPipeline(
        model=LlamaForCausalLM(config),
        tokenizer=_Tokenizer(),
        model_name_or_path="local/tiny-llama",
        controls=[control],
    )

    pipeline.steer()
    spipe = pipeline.to_spipe()

    entry = spipe.manifest["controls"][0]
    assert entry["method"] == "structural_control/fisher_drift"
    assert entry["resolved"]["method"] == "structural_control/load_lora"
    saved = spipe.save(tmp_path / "fisher-drift.spipe")
    rebuilt = type(spipe).load(saved).pipeline()
    assert type(rebuilt.structural_controls[0]).__name__ == "LoadLoRA"
