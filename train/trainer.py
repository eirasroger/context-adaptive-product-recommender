"""The training loop."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.dataset import ComparisonSetDataset, collate
from core.prepare import Prepared
from core.registry import Registry
from eval import metrics as metrics_module
from model.loss import LossWeights, composite_loss
from model.recommender import ModelConfig, Recommender
from train.config import TrainConfig


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_pointwise: float
    train_pairwise: float
    val_loss: float
    val_gap_fidelity: float
    val_band_placement: float
    learning_rate: float
    seconds: float
    by_provenance: dict[str, float] = field(default_factory=dict)

    def line(self) -> str:
        provenance = "  ".join(
            f"{key}={value:.4f}" for key, value in sorted(self.by_provenance.items())
        )
        return (
            f"epoch {self.epoch:3d}  train {self.train_loss:.5f} "
            f"(point {self.train_pointwise:.5f} pair {self.train_pairwise:.5f})  "
            f"val {self.val_loss:.5f}  gap {self.val_gap_fidelity:.5f}  "
            f"band {self.val_band_placement:.5f}  lr {self.learning_rate:.2e}  "
            f"{self.seconds:.1f}s  {provenance}"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_weights(config: TrainConfig, registry: Registry) -> LossWeights:
    return LossWeights(
        pointwise=config.loss.pointwise,
        pairwise=config.loss.pairwise,
        listwise=config.loss.listwise,
        provenance={
            key: dict(category.provenance_weights)
            for key, category in registry.categories.items()
        },
    )


def build_model(config: TrainConfig, registry: Registry) -> Recommender:
    return Recommender(
        ModelConfig.for_registry(
            registry,
            dim=config.arch.dim,
            encoder_heads=config.arch.encoder_heads,
            encoder_blocks=config.arch.encoder_blocks,
            comparator_heads=config.arch.comparator_heads,
            comparator_blocks=config.arch.comparator_blocks,
            ffn_factor=config.arch.ffn_factor,
            dropout=config.arch.dropout,
            use_family_prior=config.arch.use_family_prior,
            use_context_residual=config.arch.use_context_residual,
        )
    )


def make_loaders(
    prepared: Prepared, config: TrainConfig
) -> tuple[DataLoader, DataLoader]:
    train = ComparisonSetDataset(prepared, fold="train")
    val = ComparisonSetDataset(prepared, fold="val")
    if not len(train) or not len(val):
        raise ValueError("the snapshot has no labelled training or validation sets")
    pin = config.optim.pin_memory and config.resolved_device().startswith("cuda")
    return (
        DataLoader(
            train,
            batch_size=config.optim.batch_size,
            shuffle=True,
            collate_fn=collate,
            drop_last=False,
            pin_memory=pin,
            num_workers=config.optim.num_workers,
        ),
        DataLoader(
            val,
            batch_size=config.optim.batch_size,
            shuffle=False,
            collate_fn=collate,
            pin_memory=pin,
            num_workers=config.optim.num_workers,
        ),
    )


def _schedule(step: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay, as a multiplier on the base rate."""
    if warmup and step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model, loader, weights, device) -> tuple[float, metrics_module.MetricSet]:
    model.eval()
    total, batches = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        scores = model(batch)
        breakdown = composite_loss(
            scores,
            batch.pref,
            batch.conf,
            batch.alternative_mask,
            batch.provenance,
            batch.category_keys,
            weights,
        )
        total += float(breakdown.total)
        batches += 1
    predictions = metrics_module.predict(model, loader, device)
    return total / max(1, batches), metrics_module.compute(predictions)


def train(
    config: TrainConfig,
    registry: Registry,
    prepared: Prepared,
    on_epoch=None,
) -> tuple[Recommender, list[EpochRecord]]:
    """Train to convergence, keeping the best weights by validation loss."""
    set_seed(config.optim.seed)
    device = torch.device(config.resolved_device())

    train_loader, val_loader = make_loaders(prepared, config)
    model = build_model(config, registry).to(device)
    weights = loss_weights(config, registry)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
    )
    total_steps = max(1, config.optim.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _schedule(step, config.optim.warmup_steps, total_steps),
    )

    history: list[EpochRecord] = []
    best_loss = float("inf")
    best_state = None
    stale = 0

    for epoch in range(1, config.optim.epochs + 1):
        started = time.time()
        model.train()
        totals = {"total": 0.0, "pointwise": 0.0, "pairwise": 0.0}
        provenance_totals: dict[str, list[float]] = {}
        batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(batch)
            breakdown = composite_loss(
                scores,
                batch.pref,
                batch.conf,
                batch.alternative_mask,
                batch.provenance,
                batch.category_keys,
                weights,
            )
            breakdown.total.backward()
            if config.optim.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            optimizer.step()
            scheduler.step()

            totals["total"] += float(breakdown.total.detach())
            totals["pointwise"] += float(breakdown.pointwise)
            totals["pairwise"] += float(breakdown.pairwise)
            for key, value in breakdown.by_provenance.items():
                provenance_totals.setdefault(key, []).append(value)
            batches += 1

        val_loss, val_metrics = evaluate(model, val_loader, weights, device)
        record = EpochRecord(
            epoch=epoch,
            train_loss=totals["total"] / max(1, batches),
            train_pointwise=totals["pointwise"] / max(1, batches),
            train_pairwise=totals["pairwise"] / max(1, batches),
            val_loss=val_loss,
            val_gap_fidelity=val_metrics.gap_fidelity,
            val_band_placement=val_metrics.band_placement,
            learning_rate=optimizer.param_groups[0]["lr"],
            seconds=time.time() - started,
            by_provenance={
                key: float(np.mean(values)) for key, values in provenance_totals.items()
            },
        )
        history.append(record)
        if on_epoch is not None:
            on_epoch(record)

        if val_loss + config.optim.min_delta < best_loss:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.optim.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def write_history(path: Path, history: list[EpochRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([record.__dict__ for record in history], indent=2), encoding="utf-8"
    )
