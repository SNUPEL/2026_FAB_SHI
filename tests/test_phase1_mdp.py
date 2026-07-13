"""Phase 1 MDP/self-labeling surface tests.

These tests pin down the current learning interface before adding a full
training loop:
- Phase 1 has two decisions per block: SELECT_BLOCK and SELECT_BAY.
- The teacher label comes from the deterministic Phase 1 heuristic.
- Missing objective columns must fail, not silently become zero.
"""

from pathlib import Path
from types import SimpleNamespace
import csv
import json
import tempfile
import unittest

import torch

from Phase1.imitation import train_phase1_pointer_imitation
from Phase1.pointer_policy import Phase1PointerPolicy
from Utils.phase1.phase1_bay_balancer import LONG_CUT_PREFERRED_PHASE1_HEURISTIC
from Utils.phase1.phase1_mdp import (
    PHASE1_BAY_FEATURE_NAMES,
    PHASE1_BLOCK_FEATURE_NAMES,
    PHASE1_ENV_FEATURE_NAMES,
    write_phase1_mdp_trace_package,
)


class Phase1MDPTraceTest(unittest.TestCase):
    """Phase 1 learning data should be inspectable and deterministic."""

    def test_writes_select_block_and_select_bay_trace_package(self) -> None:
        """One block assignment should create two supervised action rows."""

        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=1200, bevel_quantity=2),
            "WO_B": self._job("WO_B", "P1::B", steel=7, cut_length=400, bevel_quantity=8),
            "WO_C": self._job("WO_C", "P1::C", steel=3, cut_length=200, bevel_quantity=1),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_phase1_mdp_trace_package(
                jobs=jobs,
                bay_ids=["22", "23", "24"],
                algorithm=LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
                output_dir=temp_dir,
            )
            trace_rows = self._read_csv(paths["trace_csv"])
            action_rows = [
                json.loads(line)
                for line in Path(paths["action_table_jsonl"]).read_text(encoding="utf-8").splitlines()
            ]
            manifest = json.loads(Path(paths["manifest_json"]).read_text(encoding="utf-8"))

        self.assertEqual(len(trace_rows), 6)
        self.assertEqual([row["phase"] for row in trace_rows[:2]], ["SELECT_BLOCK", "SELECT_BAY"])
        self.assertEqual(len(action_rows), len(trace_rows))
        self.assertEqual(manifest["decision_phases"], ["SELECT_BLOCK", "SELECT_BAY"])
        self.assertEqual(manifest["block_feature_names"], PHASE1_BLOCK_FEATURE_NAMES)
        self.assertEqual(manifest["bay_feature_names"], PHASE1_BAY_FEATURE_NAMES)
        self.assertEqual(manifest["env_feature_names"], PHASE1_ENV_FEATURE_NAMES)
        self.assertEqual(manifest["summary"]["block_count"], 3)
        self.assertEqual(trace_rows[0]["selected_block_set_id"], "P1::A")
        self.assertEqual(trace_rows[0]["candidate_count"], "3")
        self.assertEqual(action_rows[0]["selected_action_index"], 0)
        self.assertEqual(action_rows[0]["candidates"][0]["block_set_id"], "P1::A")
        self.assertIn(trace_rows[1]["selected_bay"], {"22", "23", "24"})

    def test_missing_bevel_quantity_fails_without_fallback(self) -> None:
        """BV_QTY is required for Phase 1 multi-objective/self-label traces."""

        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=100, bevel_quantity=None),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "missing_bevel_quantity"):
                write_phase1_mdp_trace_package(
                    jobs=jobs,
                    bay_ids=["22", "23", "24"],
                    algorithm=LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
                    output_dir=temp_dir,
                )

    def test_phase1_pointer_policy_scores_block_and_bay_candidates(self) -> None:
        """The Phase 1 network should match the two-stage decision shape."""

        model = Phase1PointerPolicy(
            block_feature_dim=len(PHASE1_BLOCK_FEATURE_NAMES),
            bay_feature_dim=len(PHASE1_BAY_FEATURE_NAMES),
            env_feature_dim=len(PHASE1_ENV_FEATURE_NAMES),
            hidden_dim=16,
        )
        block_features = torch.zeros((4, len(PHASE1_BLOCK_FEATURE_NAMES)), dtype=torch.float32)
        bay_features = torch.zeros((3, len(PHASE1_BAY_FEATURE_NAMES)), dtype=torch.float32)
        env_features = torch.zeros((len(PHASE1_ENV_FEATURE_NAMES),), dtype=torch.float32)

        block_logits = model.score_blocks(block_features, env_features)
        bay_logits = model.score_bays(
            bay_features=bay_features,
            env_features=env_features,
            selected_block_features=block_features[0],
        )

        self.assertEqual(tuple(block_logits.shape), (4,))
        self.assertEqual(tuple(bay_logits.shape), (3,))

    def test_phase1_imitation_training_writes_checkpoint_and_metrics(self) -> None:
        """The generated action table should train one small imitation epoch."""

        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=1200, bevel_quantity=2),
            "WO_B": self._job("WO_B", "P1::B", steel=7, cut_length=400, bevel_quantity=8),
            "WO_C": self._job("WO_C", "P1::C", steel=3, cut_length=200, bevel_quantity=1),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            trace_dir = Path(temp_dir) / "trace"
            train_dir = Path(temp_dir) / "train"
            paths = write_phase1_mdp_trace_package(
                jobs=jobs,
                bay_ids=["22", "23", "24"],
                algorithm=LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
                output_dir=trace_dir,
            )
            result = train_phase1_pointer_imitation(
                action_table_path=paths["action_table_jsonl"],
                output_dir=train_dir,
                epochs=1,
                lr=0.01,
                hidden_dim=16,
                seed=123,
            )

            self.assertTrue(Path(result["checkpoint_path"]).exists())
            self.assertTrue(Path(result["metrics_csv"]).exists())
            self.assertTrue(Path(result["summary_json"]).exists())
            self.assertEqual(result["example_count"], 6)
            self.assertGreaterEqual(result["final_accuracy"], 0.0)
            self.assertLessEqual(result["final_accuracy"], 1.0)

    @staticmethod
    def _job(
        job_id: str,
        block_set_id: str,
        steel: int,
        cut_length: float,
        bevel_quantity: int | None,
    ) -> SimpleNamespace:
        """Create the smallest Job-like object needed by Phase 1 logic."""

        return SimpleNamespace(
            job_id=job_id,
            block_set_id=block_set_id,
            steel_quantity=steel,
            cut_length=cut_length,
            bevel_quantity=bevel_quantity,
            cut_bay=None,
            source_cut_bay=None,
            allowed_bay_ids=(),
            extra={
                "source_project_no": block_set_id.split("::")[0],
                "source_block_no": block_set_id.split("::")[1],
                "source_wk_ord_no": job_id,
            },
        )

    @staticmethod
    def _read_csv(path: str) -> list[dict]:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))


if __name__ == "__main__":
    unittest.main()
