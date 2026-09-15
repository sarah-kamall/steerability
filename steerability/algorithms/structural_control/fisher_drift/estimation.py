"""Dataset collation and gradient estimators for `FisherDrift`."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


def collate_rows(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    data_collator=None,
) -> dict[str, torch.Tensor]:
    """Collate pretokenized causal-LM rows into right-padded tensors.

    Args:
        rows: Dataset rows containing `input_ids` and optional masks and labels.
        tokenizer: Tokenizer supplying the padding token id.
        max_length: Maximum retained row length.
        data_collator: Optional user collator. Its result is validated and truncated.

    Returns:
        A mapping with two-dimensional `input_ids`, `attention_mask`, and `labels`.
    """
    if data_collator is not None:
        batch = dict(data_collator(list(rows)))
        if "input_ids" not in batch:
            raise ValueError("data_collator output must contain input_ids.")
        input_ids = torch.as_tensor(batch["input_ids"], dtype=torch.long)
        attention_mask = torch.as_tensor(batch.get("attention_mask", torch.ones_like(input_ids)), dtype=torch.long)
        labels = torch.as_tensor(batch.get("labels", input_ids.clone()), dtype=torch.long)
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or labels.shape != input_ids.shape:
            raise ValueError("Collated input_ids, attention_mask, and labels must have the same two-dimensional shape.")
        return {
            "input_ids": input_ids[:, :max_length],
            "attention_mask": attention_mask[:, :max_length],
            "labels": labels[:, :max_length],
        }

    normalized = []
    for row in rows:
        if not isinstance(row, Mapping) or "input_ids" not in row:
            raise ValueError("Every dataset row must be a mapping containing input_ids.")
        input_ids = torch.as_tensor(row["input_ids"], dtype=torch.long).flatten()[:max_length]
        attention_mask = torch.as_tensor(row.get("attention_mask", torch.ones_like(input_ids)), dtype=torch.long).flatten()
        labels = torch.as_tensor(row.get("labels", input_ids.clone()), dtype=torch.long).flatten()
        attention_mask = attention_mask[: len(input_ids)]
        labels = labels[: len(input_ids)]
        if len(input_ids) < 2:
            raise ValueError("Every dataset row must contain at least two tokens.")
        if attention_mask.shape != input_ids.shape or labels.shape != input_ids.shape:
            raise ValueError("A row's input_ids, attention_mask, and labels must have matching lengths.")
        normalized.append((input_ids, attention_mask, labels))

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("FisherDrift requires a tokenizer with pad_token_id.")
    width = max(len(input_ids) for input_ids, _, _ in normalized)
    input_batch = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    mask_batch = torch.zeros((len(rows), width), dtype=torch.long)
    label_batch = torch.full((len(rows), width), -100, dtype=torch.long)
    for index, (input_ids, attention_mask, labels) in enumerate(normalized):
        length = len(input_ids)
        input_batch[index, :length] = input_ids
        mask_batch[index, :length] = attention_mask
        label_batch[index, :length] = labels
    return {"input_ids": input_batch, "attention_mask": mask_batch, "labels": label_batch}


def estimate_diagonal_fisher(
    model,
    parameters: list[torch.nn.Parameter],
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    num_samples: int,
    seed: int,
    max_length: int,
    data_collator=None,
) -> tuple[list[torch.Tensor], int]:
    """Estimate true diagonal Fisher from sampled next-token outcomes.

    One valid causal prediction context is sampled per selected reference row. The gradient
    is squared before averaging, avoiding cross terms from a batch-averaged gradient.

    Returns:
        The parameter-shaped float32 Fisher diagonal and number of sampled contexts.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    row_count = min(num_samples, len(dataset))
    indices = torch.randperm(len(dataset), generator=generator)[:row_count].tolist()
    accumulator = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in parameters]
    input_device = model.get_input_embeddings().weight.device
    contexts = 0

    model.eval()
    for row_index in indices:
        batch = collate_rows(
            [dataset[row_index]], tokenizer, max_length=max_length, data_collator=data_collator,
        )
        input_ids = batch["input_ids"].to(input_device)
        attention_mask = batch["attention_mask"].to(input_device)
        labels = batch["labels"].to(input_device)
        valid = (
            (attention_mask[:, 1:] != 0)
            & (attention_mask[:, :-1] != 0)
            & (labels[:, 1:] != -100)
        ).nonzero(as_tuple=False)
        if valid.numel() == 0:
            continue
        chosen = valid[int(torch.randint(len(valid), (1,), generator=generator).item())]
        prediction_position = int(chosen[1].item())

        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        token_logits = logits[int(chosen[0].item()), prediction_position].float()
        probabilities = token_logits.detach().softmax(dim=-1).cpu()
        sampled_token = int(torch.multinomial(probabilities, 1, generator=generator).item())
        loss = -F.log_softmax(token_logits, dim=-1)[sampled_token]
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        for diagonal, gradient in zip(accumulator, gradients):
            if gradient is not None:
                diagonal.add_(gradient.detach().float().square())
        contexts += 1

    if contexts == 0:
        raise ValueError("prior_dataset contains no valid causal prediction contexts.")
    for diagonal in accumulator:
        diagonal.div_(contexts)
    return accumulator, contexts


def compute_target_gradient(
    model,
    parameters: list[torch.nn.Parameter],
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int,
    max_length: int,
    data_collator=None,
) -> tuple[list[torch.Tensor], int]:
    """Compute the mean supervised next-token gradient over the target dataset."""
    accumulator = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in parameters]
    input_device = model.get_input_embeddings().weight.device
    token_count = 0
    model.eval()

    for start in range(0, len(dataset), batch_size):
        rows = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        batch = collate_rows(rows, tokenizer, max_length=max_length, data_collator=data_collator)
        input_ids = batch["input_ids"].to(input_device)
        attention_mask = batch["attention_mask"].to(input_device)
        labels = batch["labels"].to(input_device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
        shift_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
        valid_predictions = (attention_mask[:, 1:] != 0) & (attention_mask[:, :-1] != 0)
        shifted = labels[:, 1:].masked_fill(~valid_predictions, -100)
        shift_labels = shifted.reshape(-1)
        valid_tokens = int((shift_labels != -100).sum().item())
        if valid_tokens == 0:
            continue
        loss_sum = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="sum")
        gradients = torch.autograd.grad(loss_sum, parameters, allow_unused=True)
        for total, gradient in zip(accumulator, gradients):
            if gradient is not None:
                total.add_(gradient.detach().float())
        token_count += valid_tokens

    if token_count == 0:
        raise ValueError("train_dataset contains no supervised causal tokens.")
    for gradient in accumulator:
        gradient.div_(token_count)
    return accumulator, token_count
