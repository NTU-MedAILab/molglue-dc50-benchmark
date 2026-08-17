#!/usr/bin/env python3
"""Run the frozen post-hoc molecular-graph model-family sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rdkit
import sklearn
import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
ROUTE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_confirmatory_cpu_v1 as core  # noqa: E402
import run_post_hoc_histgradientboosting_model_family_sensitivity_v1 as hgb  # noqa: E402


ANALYSIS_LABEL = "post_hoc_graph_model_family_sensitivity"
PROTOCOL_VERSION = "post_hoc_graph_model_family_sensitivity_v1"
SEED = 260531
BOOTSTRAP_REPLICATES = 10_000

CHEM = "chemistry_graph_mpn"
FULL = "full_context_graph_mpn"
MODEL_ORDER = (CHEM, FULL)

DATA_FILE = (
    ROUTE_DIR
    / "data"
    / "processed"
    / "all_molglue_dc50_qc_train_test_standardized_context.csv"
)
PROTOCOL_FILE = (
    ROUTE_DIR
    / "docs"
    / "post_hoc_graph_model_family_sensitivity_protocol_v1.md"
)
DEFAULT_OUTPUT = (
    ROUTE_DIR
    / "reports"
    / "post_hoc_graph_model_family_sensitivity_v1"
)

ATOM_NUMBERS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53]
HYBRIDIZATIONS = [
    rdchem.HybridizationType.S,
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
BOND_CHANNELS = ("single", "double", "triple", "aromatic", "other")


@dataclass(frozen=True)
class TrainingConfig:
    hidden_dim: int = 128
    graph_dim: int = 128
    context_dim: int = 64
    n_layers: int = 3
    dropout: float = 0.15
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 120
    patience: int = 15
    smooth_l1_beta: float = 0.5
    gradient_clip_norm: float = 5.0
    inner_folds: int = 5


FORMAL_CONFIG = TrainingConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--smoke-epochs", type=int, default=2)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def select_device(requested: str, smoke: bool) -> torch.device:
    if requested == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected = requested
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if not smoke and selected != "cuda":
        raise RuntimeError("The frozen formal graph protocol requires a CUDA device")
    return torch.device(selected)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def one_hot_unknown(value: Any, choices: Iterable[Any]) -> list[float]:
    choices_list = list(choices)
    return [float(value == choice) for choice in choices_list] + [
        float(value not in choices_list)
    ]


def atom_features(atom: Chem.Atom) -> list[float]:
    return (
        one_hot_unknown(atom.GetAtomicNum(), ATOM_NUMBERS)
        + one_hot_unknown(min(int(atom.GetDegree()), 6), range(6))
        + one_hot_unknown(min(int(atom.GetTotalNumHs()), 5), range(5))
        + one_hot_unknown(max(-3, min(3, int(atom.GetFormalCharge()))), range(-2, 3))
        + one_hot_unknown(atom.GetHybridization(), HYBRIDIZATIONS)
        + [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.HasProp("_ChiralityPossible")),
        ]
    )


def bond_channel(bond: Chem.Bond) -> int:
    bond_type = bond.GetBondType()
    if bond_type == rdchem.BondType.SINGLE:
        return 0
    if bond_type == rdchem.BondType.DOUBLE:
        return 1
    if bond_type == rdchem.BondType.TRIPLE:
        return 2
    if bond_type == rdchem.BondType.AROMATIC:
        return 3
    return 4


def smiles_to_graph(smiles: str) -> dict[str, np.ndarray]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"RDKit could not build a non-empty graph: {smiles}")
    x = np.asarray([atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)
    adjacency = np.zeros(
        (len(BOND_CHANNELS), mol.GetNumAtoms(), mol.GetNumAtoms()),
        dtype=np.float32,
    )
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        channel = bond_channel(bond)
        adjacency[channel, begin, end] = 1.0
        adjacency[channel, end, begin] = 1.0
    return {"x": x, "adjacency": adjacency}


def build_graphs(smiles_values: Iterable[str]) -> tuple[list[dict[str, np.ndarray]], pd.DataFrame]:
    graphs: list[dict[str, np.ndarray]] = []
    audit: list[dict[str, Any]] = []
    for row_index, smiles in enumerate(smiles_values):
        graph = smiles_to_graph(str(smiles))
        graphs.append(graph)
        audit.append(
            {
                "row_index": row_index,
                "n_atoms": int(graph["x"].shape[0]),
                "n_undirected_bonds": int(graph["adjacency"].sum() / 2),
                "atom_feature_dim": int(graph["x"].shape[1]),
            }
        )
    return graphs, pd.DataFrame(audit)


class GraphDataset(Dataset):
    def __init__(
        self,
        graphs: list[dict[str, np.ndarray]],
        context: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
    ):
        self.graphs = graphs
        self.context = context
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
        idx = int(self.indices[position])
        return self.graphs[idx], self.context[idx], float(self.y[idx])


def collate_graph_batch(
    batch: list[tuple[dict[str, np.ndarray], np.ndarray, float]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(batch)
    max_atoms = max(item[0]["x"].shape[0] for item in batch)
    atom_dim = batch[0][0]["x"].shape[1]
    context_dim = batch[0][1].shape[0]
    atom_x = np.zeros((batch_size, max_atoms, atom_dim), dtype=np.float32)
    adjacency = np.zeros(
        (batch_size, len(BOND_CHANNELS), max_atoms, max_atoms),
        dtype=np.float32,
    )
    mask = np.zeros((batch_size, max_atoms), dtype=np.float32)
    context = np.zeros((batch_size, context_dim), dtype=np.float32)
    y = np.zeros((batch_size, 1), dtype=np.float32)
    for position, (graph, context_row, target) in enumerate(batch):
        n_atoms = graph["x"].shape[0]
        atom_x[position, :n_atoms] = graph["x"]
        adjacency[position, :, :n_atoms, :n_atoms] = graph["adjacency"]
        mask[position, :n_atoms] = 1.0
        context[position] = context_row
        y[position, 0] = target
    return (
        torch.from_numpy(atom_x),
        torch.from_numpy(adjacency),
        torch.from_numpy(mask),
        torch.from_numpy(context),
        torch.from_numpy(y),
    )


class BondMessagePassingEncoder(nn.Module):
    def __init__(self, atom_dim: int, config: TrainingConfig):
        super().__init__()
        self.input_projection = nn.Linear(atom_dim, config.hidden_dim)
        self.self_layers = nn.ModuleList(
            [nn.Linear(config.hidden_dim, config.hidden_dim) for _ in range(config.n_layers)]
        )
        self.bond_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
                        for _ in BOND_CHANNELS
                    ]
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(config.hidden_dim) for _ in range(config.n_layers)]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.graph_dim),
            nn.LayerNorm(config.graph_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        atom_x: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.input_projection(atom_x))
        h = h * mask.unsqueeze(-1)
        for self_layer, edge_layers, norm in zip(
            self.self_layers, self.bond_layers, self.norms
        ):
            update = self_layer(h)
            for channel, edge_layer in enumerate(edge_layers):
                local_adj = adjacency[:, channel]
                degree = local_adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
                aggregate = torch.bmm(local_adj, h) / degree
                update = update + edge_layer(aggregate)
            update = torch.nn.functional.gelu(update)
            h = norm(h + self.dropout(update))
            h = h * mask.unsqueeze(-1)
        masked = h * mask.unsqueeze(-1)
        mean_pool = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        max_pool = masked.masked_fill(mask.unsqueeze(-1).eq(0), -1e9).max(dim=1)[0]
        max_pool = torch.where(max_pool < -1e8, torch.zeros_like(max_pool), max_pool)
        return self.output(torch.cat([mean_pool, max_pool], dim=1))


class GraphRegressor(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        input_context_dim: int,
        config: TrainingConfig,
    ):
        super().__init__()
        self.graph_encoder = BondMessagePassingEncoder(atom_dim, config)
        self.has_context = input_context_dim > 0
        if self.has_context:
            self.context_projection: nn.Module | None = nn.Sequential(
                nn.Linear(input_context_dim, config.context_dim),
                nn.LayerNorm(config.context_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            fusion_dim = config.graph_dim + config.context_dim
        else:
            self.context_projection = None
            fusion_dim = config.graph_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        atom_x: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        graph = self.graph_encoder(atom_x, adjacency, mask)
        if self.has_context:
            if self.context_projection is None:
                raise RuntimeError("Context projection is unexpectedly absent")
            graph = torch.cat([graph, self.context_projection(context)], dim=1)
        return self.head(graph)


def predict(
    model: GraphRegressor,
    dataset: GraphDataset,
    device: torch.device,
    batch_size: int,
    y_mean: float,
    y_std: float,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_graph_batch,
    )
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for atom_x, adjacency, mask, context, _ in loader:
            prediction = model(
                atom_x.to(device),
                adjacency.to(device),
                mask.to(device),
                context.to(device),
            )
            output.append(prediction.detach().cpu().numpy().reshape(-1))
    values = np.concatenate(output).astype(np.float64)
    return values * y_std + y_mean


def inner_fit_validation_split(
    train_idx: np.ndarray,
    scaffold_ids: np.ndarray,
    seed: int,
    n_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    splits = core.balanced_group_splits(
        np.asarray(train_idx, dtype=np.int64),
        scaffold_ids,
        n_splits=n_folds,
        seed=seed,
    )
    fit_idx, validation_idx = splits[0]
    return (
        np.asarray(fit_idx, dtype=np.int64),
        np.asarray(validation_idx, dtype=np.int64),
    )


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        return float("nan")
    result = spearmanr(y_true, y_pred)
    value = (
        result.statistic
        if hasattr(result, "statistic")
        else result.correlation
    )
    return float(value)


def validation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    rho = safe_spearman(y_true, y_pred)
    rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
    return rho, rmse


def fit_predict_one(
    *,
    graphs: list[dict[str, np.ndarray]],
    rows: pd.DataFrame,
    y: np.ndarray,
    scaffold_ids: np.ndarray,
    split: hgb.EvaluationSplit,
    model_id: str,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    seed = int(split.seed)
    set_seed(seed)
    fit_idx, validation_idx = inner_fit_validation_split(
        split.train_idx,
        scaffold_ids,
        seed + 100_003,
        config.inner_folds,
    )
    test_idx = np.asarray(split.test_idx, dtype=np.int64)

    if set(fit_idx).intersection(validation_idx):
        raise RuntimeError("Inner fit/validation row leakage")
    if set(fit_idx).intersection(test_idx) or set(validation_idx).intersection(test_idx):
        raise RuntimeError("Inner training/test row leakage")
    fit_scaffolds = set(scaffold_ids[fit_idx].astype(str))
    validation_scaffolds = set(scaffold_ids[validation_idx].astype(str))
    if fit_scaffolds.intersection(validation_scaffolds):
        raise RuntimeError("Inner fit/validation scaffold leakage")

    if model_id == FULL:
        encoder = core.make_one_hot_encoder()
        contexts = core.context_frame(rows)
        encoder.fit(contexts.iloc[fit_idx])
        context_all = encoder.transform(contexts).astype(np.float32)
        context_dim = int(context_all.shape[1])
    elif model_id == CHEM:
        context_all = np.zeros((len(rows), 0), dtype=np.float32)
        context_dim = 0
    else:
        raise ValueError(f"Unknown graph model: {model_id}")

    y_mean = float(np.mean(y[fit_idx]))
    y_std = float(np.std(y[fit_idx]))
    if not math.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0
    y_z = ((y - y_mean) / y_std).astype(np.float32)

    fit_dataset = GraphDataset(graphs, context_all, y_z, fit_idx)
    validation_dataset = GraphDataset(graphs, context_all, y_z, validation_idx)
    test_dataset = GraphDataset(graphs, context_all, y_z, test_idx)
    generator = torch.Generator()
    generator.manual_seed(seed)
    fit_loader = DataLoader(
        fit_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_graph_batch,
        generator=generator,
    )

    model = GraphRegressor(
        atom_dim=int(graphs[0]["x"].shape[1]),
        input_context_dim=context_dim,
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.SmoothL1Loss(beta=config.smooth_l1_beta)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_rho = -float("inf")
    best_rmse = float("inf")
    stalled = 0
    epochs_trained = 0
    start = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for atom_x, adjacency, mask, context, target in fit_loader:
            optimizer.zero_grad()
            predicted = model(
                atom_x.to(device),
                adjacency.to(device),
                mask.to(device),
                context.to(device),
            )
            loss = criterion(predicted, target.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss: {split.heldout_group} {model_id}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()

        validation_prediction = predict(
            model,
            validation_dataset,
            device,
            config.batch_size,
            y_mean,
            y_std,
        )
        validation_rho, validation_rmse = validation_metrics(
            y[validation_idx],
            validation_prediction,
        )
        comparable_rho = (
            validation_rho if math.isfinite(validation_rho) else -float("inf")
        )
        improved = (
            comparable_rho > best_rho + 1e-12
            or (
                math.isclose(comparable_rho, best_rho, abs_tol=1e-12)
                and validation_rmse < best_rmse - 1e-12
            )
        )
        if improved or best_state is None:
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            best_epoch = epoch
            best_rho = comparable_rho
            best_rmse = validation_rmse
            stalled = 0
        else:
            stalled += 1
        epochs_trained = epoch
        if stalled >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Training ended without a finite checkpoint")
    model.load_state_dict(best_state)
    prediction = predict(
        model,
        test_dataset,
        device,
        config.batch_size,
        y_mean,
        y_std,
    )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Non-finite prediction from {model_id}")

    audit = {
        "analysis_label": ANALYSIS_LABEL,
        "split_regime": split.split_regime,
        "protocol": split.protocol,
        "heldout_group": split.heldout_group,
        "split_order": split.split_order,
        "model_id": model_id,
        "seed": seed,
        "n_outer_train": int(len(split.train_idx)),
        "n_inner_fit": int(len(fit_idx)),
        "n_inner_validation": int(len(validation_idx)),
        "n_test": int(len(test_idx)),
        "n_inner_fit_scaffolds": int(len(fit_scaffolds)),
        "n_inner_validation_scaffolds": int(len(validation_scaffolds)),
        "inner_fit_validation_row_overlap": 0,
        "inner_fit_validation_scaffold_overlap": 0,
        "inner_train_test_row_overlap": 0,
        "context_dim": context_dim,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "best_validation_spearman": (
            best_rho if math.isfinite(best_rho) else float("nan")
        ),
        "best_validation_rmse": best_rmse,
        "fit_seconds": time.perf_counter() - start,
    }
    return prediction, audit


def run_splits(
    splits: list[hgb.EvaluationSplit],
    graphs: list[dict[str, np.ndarray]],
    rows: pd.DataFrame,
    y: np.ndarray,
    scaffold_ids: np.ndarray,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for split_position, split in enumerate(splits, start=1):
        print(
            f"[FIT] regime={split.split_regime} protocol={split.protocol} "
            f"group={split.heldout_group} split={split_position}/{len(splits)}",
            flush=True,
        )
        for model_id in MODEL_ORDER:
            prediction, training_audit = fit_predict_one(
                graphs=graphs,
                rows=rows,
                y=y,
                scaffold_ids=scaffold_ids,
                split=split,
                model_id=model_id,
                config=config,
                device=device,
            )
            audit_records.append(training_audit)
            for local_position, row_idx in enumerate(split.test_idx):
                prediction_records.append(
                    {
                        "analysis_label": ANALYSIS_LABEL,
                        "split_regime": split.split_regime,
                        "protocol": split.protocol,
                        "heldout_group": split.heldout_group,
                        "split_order": split.split_order,
                        "row_index": int(row_idx),
                        "qc_id": str(rows.iloc[row_idx]["qc_id"]),
                        "model_id": model_id,
                        "y_true": float(y[row_idx]),
                        "y_pred": float(prediction[local_position]),
                        "canonical_smiles": str(
                            rows.iloc[row_idx]["canonical_smiles"]
                        ),
                        "scaffold_id": str(scaffold_ids[row_idx]),
                        "source_database": str(
                            rows.iloc[row_idx]["source_database"]
                        ),
                        "target_protein": str(
                            rows.iloc[row_idx]["target_protein"]
                        ),
                        "recruiting_protein": str(
                            rows.iloc[row_idx]["recruiting_protein"]
                        ),
                        "cell_line": str(rows.iloc[row_idx]["cell_line"]),
                    }
                )
        print(
            f"[FIT] completed={split_position}/{len(splits)} "
            f"elapsed={time.perf_counter() - start:.1f}s",
            flush=True,
        )
    return pd.DataFrame(prediction_records), pd.DataFrame(audit_records)


def build_results_brief(
    output_dir: Path,
    internal_metrics: pd.DataFrame,
    ood_aggregate: pd.DataFrame,
    bootstrap: pd.DataFrame,
    smoke: bool,
) -> None:
    contrast = bootstrap[bootstrap["record_type"] == "contrast"].copy()
    internal = contrast[contrast["split_regime"] == "internal"].iloc[0]
    model_lines = []
    for row in internal_metrics.to_dict("records"):
        model_lines.append(
            f"| internal scaffold | {row['model_id']} | "
            f"{row['spearman']:.4f} | {row['rmse']:.4f} |"
        )
    for row in ood_aggregate.to_dict("records"):
        rho = (
            "NA"
            if not math.isfinite(float(row["domain_macro_spearman"]))
            else f"{row['domain_macro_spearman']:.4f}"
        )
        model_lines.append(
            f"| {row['split_regime']} / {row['protocol']} | "
            f"{row['model_id']} | {rho} | {row['domain_macro_rmse']:.4f} |"
        )

    contrast_lines = [
        "| internal scaffold | "
        f"{internal['delta_spearman_observed']:.4f} "
        f"[{internal['delta_spearman_ci_low']:.4f}, "
        f"{internal['delta_spearman_ci_high']:.4f}] | "
        f"{internal['delta_rmse_observed']:.4f} "
        f"[{internal['delta_rmse_ci_low']:.4f}, "
        f"{internal['delta_rmse_ci_high']:.4f}] |"
    ]
    for row in contrast[contrast["split_regime"] != "internal"].to_dict("records"):
        rho = row["delta_domain_macro_spearman_observed"]
        rho_text = (
            "不可估"
            if not math.isfinite(float(rho))
            else (
                f"{rho:.4f} "
                f"[{row['delta_domain_macro_spearman_ci_low']:.4f}, "
                f"{row['delta_domain_macro_spearman_ci_high']:.4f}]"
            )
        )
        contrast_lines.append(
            f"| {row['split_regime']} / {row['protocol']} | {rho_text} | "
            f"{row['delta_domain_macro_rmse_observed']:.4f} "
            f"[{row['delta_domain_macro_rmse_ci_low']:.4f}, "
            f"{row['delta_domain_macro_rmse_ci_high']:.4f}] |"
        )

    decision = (
        "这是代码、GPU、拆分和输出路径 smoke test，不作科学判定。"
        if smoke
        else (
            "该表是冻结的后验独立图模型敏感性结果；应与原有 ExtraTrees 和 "
            "HistGradientBoosting 证据并列解释，不得作为模型筛选结果。"
        )
    )
    brief = f"""# 独立分子图模型后验敏感性结果

## 身份

- 分析：`{ANALYSIS_LABEL}`；
- 固定 bond-aware message-passing graph learner；
- chemistry-only 与 full-context 为严格配对模型；
- 该结果不是 confirmatory，不报告 p 值；
- smoke test：`{str(smoke).lower()}`。

## 模型结果

| 场景 | 模型 | Spearman / domain-macro Spearman | RMSE / domain-macro RMSE |
|---|---|---:|---:|
{os.linesep.join(model_lines)}

## Full − chemistry-only 配对对比

正的 `Delta Spearman` 和 `Delta RMSE` 有利于 full。

| 场景 | Delta Spearman（95% CI） | Delta RMSE（95% CI） |
|---|---:|---:|
{os.linesep.join(contrast_lines)}

## 判定

{decision}

## 边界

- 本分析检验独立图表示与学习器下的评价模式，不是 leaderboard；
- 固定 source/target 域不是从更大总体随机抽样；
- 结果不建立绝对 unseen-target 泛化、机制、因果、前瞻验证或部署主张。
"""
    (output_dir / "results_brief_zh.md").write_text(brief, encoding="utf-8")


def artifact_inventory(output_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "artifact_sha256.csv":
            continue
        records.append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.smoke and args.output_dir == DEFAULT_OUTPUT:
        output_dir = Path(f"{DEFAULT_OUTPUT}_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    if list(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty; refusing overwrite: {output_dir}"
        )

    data_hash = sha256_file(DATA_FILE)
    if data_hash != core.FROZEN_DATA_SHA256:
        raise RuntimeError("Frozen data hash mismatch")
    device = select_device(args.device, bool(args.smoke))
    config = FORMAL_CONFIG
    if args.smoke:
        if args.smoke_epochs < 1:
            raise ValueError("--smoke-epochs must be positive")
        config = TrainingConfig(
            **{
                **asdict(FORMAL_CONFIG),
                "max_epochs": int(args.smoke_epochs),
                "patience": int(args.smoke_epochs),
            }
        )

    hgb.ANALYSIS_LABEL = ANALYSIS_LABEL
    hgb.CHEM = CHEM
    hgb.FULL = FULL
    hgb.MODEL_ORDER = MODEL_ORDER
    hgb.safe_spearman = safe_spearman

    started = time.time()
    rows, context_audit = core.load_rows(DATA_FILE)
    y = rows["pDC50"].to_numpy(dtype=np.float64)
    scaffold_ids = core.build_scaffold_ids(rows["canonical_smiles"])
    graphs, graph_audit = build_graphs(rows["canonical_smiles"])
    internal_splits = hgb.build_internal_splits(
        rows, scaffold_ids, bool(args.smoke)
    )
    ood_splits, strict_splits, strict_audit = hgb.build_ood_and_strict_splits(
        rows, scaffold_ids, bool(args.smoke)
    )

    internal_predictions, internal_training = run_splits(
        internal_splits,
        graphs,
        rows,
        y,
        scaffold_ids,
        config,
        device,
    )
    ood_predictions, ood_training = run_splits(
        [*ood_splits, *strict_splits],
        graphs,
        rows,
        y,
        scaffold_ids,
        config,
        device,
    )
    predictions = pd.concat(
        [internal_predictions, ood_predictions], ignore_index=True
    )
    training_audit = pd.concat(
        [internal_training, ood_training], ignore_index=True
    )
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    training_audit.to_csv(output_dir / "training_audit.csv", index=False)
    context_audit.to_csv(
        output_dir / "context_sanitization_audit.csv", index=False
    )
    graph_audit.to_csv(output_dir / "graph_construction_audit.csv", index=False)
    strict_audit.to_csv(output_dir / "strict_split_audit.csv", index=False)

    averaged, internal_metrics = hgb.summarize_internal(internal_predictions)
    averaged.to_csv(
        output_dir / "internal_repeat_averaged_predictions.csv", index=False
    )
    internal_metrics.to_csv(output_dir / "internal_metrics.csv", index=False)
    local_domain_metrics = hgb.domain_metrics(ood_predictions)
    local_domain_metrics.to_csv(
        output_dir / "ood_domain_metrics.csv", index=False
    )
    ood_aggregate = hgb.aggregate_ood(
        ood_predictions, local_domain_metrics
    )
    ood_aggregate.to_csv(
        output_dir / "ood_aggregate_metrics.csv", index=False
    )

    n_bootstrap = 100 if args.smoke else BOOTSTRAP_REPLICATES
    bootstrap_frames = [
        hgb.bootstrap_internal(averaged, n_bootstrap, SEED + 700_000)
    ]
    for regime_position, regime in enumerate(
        ("domain_plus_compound_cold", "strict_domain_plus_scaffold_cold"),
        start=1,
    ):
        for protocol_position, protocol in enumerate(
            ("source_ood", "target_ood"), start=1
        ):
            bootstrap_frames.append(
                hgb.bootstrap_ood(
                    ood_predictions,
                    regime,
                    protocol,
                    n_bootstrap,
                    SEED
                    + 800_000
                    + regime_position * 10_000
                    + protocol_position * 1_000,
                )
            )
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True, sort=False)
    bootstrap.loc[
        bootstrap["record_type"] == "contrast", "contrast_id"
    ] = "full_vs_chemistry_graph_mpn"
    bootstrap.to_csv(
        output_dir / "paired_global_scaffold_bootstrap.csv", index=False
    )
    build_results_brief(
        output_dir,
        internal_metrics,
        ood_aggregate,
        bootstrap,
        bool(args.smoke),
    )

    expected_internal = sum(
        len(split.test_idx) * len(MODEL_ORDER) for split in internal_splits
    )
    expected_ood = sum(
        len(split.test_idx) * len(MODEL_ORDER)
        for split in [*ood_splits, *strict_splits]
    )
    checks = {
        "data_hash_match": True,
        "all_predictions_finite": bool(
            np.isfinite(predictions["y_pred"]).all()
        ),
        "internal_prediction_count": (
            len(internal_predictions) == expected_internal
        ),
        "ood_prediction_count": len(ood_predictions) == expected_ood,
        "strict_zero_row_overlap": bool(
            (strict_audit["train_test_row_overlap"] == 0).all()
        ),
        "strict_zero_compound_overlap": bool(
            (strict_audit["train_test_smiles_overlap"] == 0).all()
        ),
        "strict_zero_scaffold_overlap": bool(
            (strict_audit["train_test_scaffold_overlap"] == 0).all()
        ),
        "inner_zero_row_overlap": bool(
            (training_audit["inner_fit_validation_row_overlap"] == 0).all()
            and (training_audit["inner_train_test_row_overlap"] == 0).all()
        ),
        "inner_zero_scaffold_overlap": bool(
            (
                training_audit["inner_fit_validation_scaffold_overlap"]
                == 0
            ).all()
        ),
        "all_graphs_nonempty": bool((graph_audit["n_atoms"] > 0).all()),
    }
    qa = {
        "analysis_label": ANALYSIS_LABEL,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "smoke": bool(args.smoke),
        "checks": checks,
    }
    write_json(output_dir / "qa_summary.json", qa)
    if qa["status"] != "PASS":
        raise RuntimeError(f"QA failure: {checks}")

    cuda_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_label": ANALYSIS_LABEL,
        "status": "complete",
        "confirmatory": False,
        "smoke": bool(args.smoke),
        "gpu_used": device.type == "cuda",
        "device": str(device),
        "cuda_device_name": cuda_name,
        "data_file": str(DATA_FILE),
        "data_sha256": data_hash,
        "script_sha256": sha256_file(Path(__file__)),
        "protocol_document": str(PROTOCOL_FILE),
        "protocol_document_sha256": sha256_file(PROTOCOL_FILE),
        "n_rows": len(rows),
        "n_unique_smiles": rows["canonical_smiles"].nunique(),
        "n_scaffolds": len(np.unique(scaffold_ids)),
        "models": list(MODEL_ORDER),
        "graph": {
            "atom_feature_dim": int(graphs[0]["x"].shape[1]),
            "bond_channels": list(BOND_CHANNELS),
            "max_atoms": int(graph_audit["n_atoms"].max()),
        },
        "training_config": asdict(config),
        "formal_training_config": asdict(FORMAL_CONFIG),
        "no_hyperparameter_selection": True,
        "splits": {
            "internal": len(internal_splits),
            "domain_plus_compound_cold": len(ood_splits),
            "strict_domain_plus_scaffold_cold": len(strict_splits),
        },
        "bootstrap_replicates": n_bootstrap,
        "seed": SEED,
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "rdkit": rdkit.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "visible_cpu_count": os.cpu_count(),
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    artifact_inventory(output_dir).to_csv(
        output_dir / "artifact_sha256.csv", index=False
    )
    print(
        f"[DONE] status=complete smoke={args.smoke} "
        f"elapsed={manifest['elapsed_seconds']:.1f}s output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
