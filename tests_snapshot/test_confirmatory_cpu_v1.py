from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_confirmatory_cpu_v1.py"
)
SPEC = importlib.util.spec_from_file_location("confirmatory_cpu_v1", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ConfirmatoryCpuV1Tests(unittest.TestCase):
    def test_balanced_splits_are_group_disjoint_and_deterministic(self) -> None:
        indices = np.arange(12, dtype=np.int64)
        groups = np.asarray(
            ["a", "a", "b", "c", "c", "d", "e", "e", "f", "g", "h", "i"],
            dtype=object,
        )
        first = MODULE.balanced_group_splits(indices, groups, 3, seed=17)
        second = MODULE.balanced_group_splits(indices, groups, 3, seed=17)
        seen: list[int] = []
        for (fit_idx, eval_idx), (_, repeated_eval_idx) in zip(first, second):
            self.assertTrue(np.array_equal(eval_idx, repeated_eval_idx))
            self.assertFalse(set(groups[fit_idx]).intersection(groups[eval_idx]))
            seen.extend(eval_idx.tolist())
        self.assertEqual(sorted(seen), indices.tolist())

    def test_knn_uses_one_median_label_per_training_compound(self) -> None:
        rows = pd.DataFrame(
            {"canonical_smiles": ["compound_a", "compound_a", "compound_b", "eval"]}
        )
        fingerprints = np.asarray(
            [[1, 0], [1, 0], [0, 1], [1, 0]],
            dtype=np.uint8,
        )
        labels = np.asarray([0.0, 10.0, 4.0, -999.0], dtype=np.float32)
        prediction = MODULE.morgan_knn_predict(
            rows,
            fingerprints,
            labels,
            fit_idx=np.asarray([0, 1, 2], dtype=np.int64),
            eval_idx=np.asarray([3], dtype=np.int64),
            k=1,
            power=1.0,
        )
        self.assertAlmostEqual(float(prediction[0]), 5.0)

    def test_descriptor_variance_uses_float64_without_overflow(self) -> None:
        descriptors = np.asarray(
            [
                [1.0e20, 1.0],
                [2.0e20, 1.0],
                [3.0e20, 1.0],
                [4.0e20, 1.0],
            ],
            dtype=np.float64,
        )
        fit, evaluation, metadata = MODULE.prepare_rdkit_fold(
            descriptors,
            fit_idx=np.asarray([0, 1, 2], dtype=np.int64),
            eval_idx=np.asarray([3], dtype=np.int64),
        )
        self.assertEqual(fit.shape, (3, 1))
        self.assertEqual(evaluation.shape, (1, 1))
        self.assertEqual(metadata["rdkit_kept"], 1)
        self.assertTrue(np.isfinite(np.var(fit, axis=0)).all())

    def test_repeat_average_precedes_primary_metrics_and_bootstrap(self) -> None:
        records: list[dict[str, object]] = []
        truth = [0.0, 1.0, 2.0, 3.0]
        models = {
            "full_context_extra_trees": truth,
            "context_ridge": list(reversed(truth)),
        }
        for repeat in (1, 2):
            for model_id, predictions in models.items():
                for row_index, (y_true, y_pred) in enumerate(
                    zip(truth, predictions)
                ):
                    records.append(
                        {
                            "protocol": "scaffold",
                            "repeat": repeat,
                            "outer_fold": row_index + 1,
                            "row_index": row_index,
                            "qc_id": f"q{row_index}",
                            "model_id": model_id,
                            "param_id": "fixed",
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "canonical_smiles": f"c{row_index}",
                            "scaffold_id": f"s{row_index}",
                            "source_database": "source",
                            "recruiting_protein": "recruiter",
                            "target_protein": "target",
                            "cell_line": "cell",
                            "max_train_tanimoto": 0.5,
                            "scaffold_seen_in_train": False,
                            "target_train_rows": 3,
                            "target_train_unique_smiles": 3,
                            "recruiter_target_train_unique_smiles": 3,
                            "source_seen_in_train": True,
                        }
                    )
        raw = pd.DataFrame(records)
        averaged = MODULE.average_cross_fitted_predictions(raw)
        self.assertEqual(len(averaged), 8)
        self.assertTrue((averaged["n_oof_predictions"] == 2).all())

        metrics = MODULE.metrics_on_repeat_averaged_predictions(averaged)
        reference = metrics[
            metrics["model_id"] == "full_context_extra_trees"
        ].iloc[0]
        self.assertAlmostEqual(float(reference["spearman"]), 1.0)

        bootstrap = MODULE.cluster_bootstrap_summary(
            averaged,
            n_bootstrap=100,
            seed=31,
        )
        comparison = bootstrap[bootstrap["model_id"] == "context_ridge"].iloc[0]
        self.assertAlmostEqual(
            float(comparison["delta_spearman_observed"]),
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
