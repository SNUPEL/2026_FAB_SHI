"""run_manifest.json 계약을 검증한다.

여러 사람이 나눠 실행한 run을 나중에 합쳐 비교하려면, 각 run이 "어떤 코드/어떤 문제집합/
어떤 실험 arm"이었는지 스스로 증명해야 한다. 이 테스트는 그 증명 파일의 필드 완전성과
불일치 시 실패 계약을 고정한다.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from Utils.learning.run_manifest import (
    RUN_MANIFEST_HISTORY_FILENAME,
    RUN_MANIFEST_SCHEMA,
    canonical_sha256,
    cli_manifest_fields,
    code_fingerprint,
    read_run_manifest,
    summarize_validation_contract,
    write_run_manifest,
)


def _cli_fields(**overrides) -> dict:
    args = SimpleNamespace(
        experiment_id="exp_alpha",
        arm_group="phase2_temperature",
        arm_label="anneal",
        owner="alice",
        run_note="first arm",
        seed=7,
        episodes=20,
        output_dir="output/x",
        temperature=1.5,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return cli_manifest_fields(
        args,
        command="phase2-train-batch-machine-self-labeling",
        phase="phase2",
    )


def _contract(seed: int = 500_000) -> dict:
    return {
        "schema": "phase2_validation_grid_v1",
        "problems": [
            {
                "problem_id": "M010_T01",
                "block_count": 10,
                "distribution_type": 1,
                "generation_seed": seed,
                "jobs_sha256": "a" * 64,
            },
            {
                "problem_id": "M010_T02",
                "block_count": 10,
                "distribution_type": 2,
                "generation_seed": seed + 1,
                "jobs_sha256": "b" * 64,
            },
        ],
    }


class RunManifestTests(unittest.TestCase):
    def test_manifest_records_identity_code_env_data_and_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_run_manifest(
                temp_dir,
                cli_fields=_cli_fields(),
                run_spec={"run_spec_schema_version": "phase2_run_spec_v4_event_clock"},
                run_spec_source="phase2_run_spec",
                validation={
                    "main": summarize_validation_contract(_contract(), fixed_grid=True, seed=0),
                    "validation_temperature": 1.0,
                },
                extra={"device": "cpu", "seed": 0},
            )
            manifest = json.loads(Path(path).read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema"], RUN_MANIFEST_SCHEMA)
        self.assertEqual(manifest["identity"]["experiment_id"], "exp_alpha")
        self.assertEqual(manifest["identity"]["arm_label"], "anneal")
        self.assertEqual(manifest["identity"]["owner"], "alice")
        self.assertTrue(manifest["identity"]["created_at_utc"])
        self.assertEqual(manifest["invocation"]["phase"], "phase2")
        self.assertEqual(manifest["invocation"]["resolved_args"]["seed"], 7)
        self.assertIn("digest", manifest["code"]["code_fingerprint"])
        self.assertIn("dirty", manifest["code"]["git"])
        self.assertTrue(manifest["env"]["python"])
        self.assertEqual(
            set(manifest["data"]["sources"]),
            {"blocks", "work_orders"},
        )
        self.assertEqual(manifest["contract"]["run_spec_source"], "phase2_run_spec")
        main_validation = manifest["validation"]["main"]
        self.assertTrue(main_validation["fixed_grid"])
        self.assertEqual(main_validation["problem_count"], 2)
        self.assertEqual(main_validation["block_sizes"], [10])
        self.assertEqual(main_validation["type_count"], 2)
        self.assertEqual(len(main_validation["contract_sha256"]), 64)

    def test_owner_defaults_to_environment_user_when_omitted(self) -> None:
        fields = _cli_fields(owner="")
        self.assertTrue(isinstance(fields["owner"], str))

    def test_rerun_with_different_arm_label_fails_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_run_manifest(
                temp_dir,
                cli_fields=_cli_fields(),
                run_spec=None,
                run_spec_source="phase2_run_spec",
                validation={"main": summarize_validation_contract(None, fixed_grid=False)},
            )
            with self.assertRaises(RuntimeError):
                write_run_manifest(
                    temp_dir,
                    cli_fields=_cli_fields(arm_label="constant"),
                    run_spec=None,
                    run_spec_source="phase2_run_spec",
                    validation={"main": summarize_validation_contract(None, fixed_grid=False)},
                )

    def test_resume_keeps_identity_and_appends_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_run_manifest(
                temp_dir,
                cli_fields=_cli_fields(),
                run_spec=None,
                run_spec_source="phase2_run_spec",
                validation={"main": summarize_validation_contract(_contract(), fixed_grid=True)},
                extra={"resumed_from_episode": 0},
            )
            write_run_manifest(
                temp_dir,
                cli_fields=_cli_fields(),
                run_spec=None,
                run_spec_source="phase2_run_spec",
                validation={"main": summarize_validation_contract(_contract(), fixed_grid=True)},
                extra={"resumed_from_episode": 1000},
            )
            history_path = Path(temp_dir) / RUN_MANIFEST_HISTORY_FILENAME
            history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
            current = read_run_manifest(temp_dir)

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["extra"]["resumed_from_episode"], 0)
        self.assertEqual(len(history[0]["previous_manifest_sha256"]), 64)
        self.assertEqual(current["extra"]["resumed_from_episode"], 1000)

    def test_read_run_manifest_rejects_missing_and_wrong_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(RuntimeError):
                read_run_manifest(temp_dir)
            (Path(temp_dir) / "run_manifest.json").write_text(
                json.dumps({"schema": "something_else"}), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                read_run_manifest(temp_dir)

    def test_code_fingerprint_is_stable_and_changes_with_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "Phase2").mkdir()
            (root / "main.py").write_text("print('a')\n", encoding="utf-8")
            (root / "Phase2" / "merged.py").write_text("x = 1\n", encoding="utf-8")
            first = code_fingerprint(root)
            self.assertEqual(first["digest"], code_fingerprint(root)["digest"])
            self.assertEqual(first["file_count"], 2)
            (root / "Phase2" / "merged.py").write_text("x = 2\n", encoding="utf-8")
            self.assertNotEqual(first["digest"], code_fingerprint(root)["digest"])

    def test_contract_sha_differs_when_validation_problems_differ(self) -> None:
        same = summarize_validation_contract(_contract(), fixed_grid=True)
        other = summarize_validation_contract(_contract(seed=900_000), fixed_grid=True)
        self.assertEqual(
            same["contract_sha256"],
            summarize_validation_contract(_contract(), fixed_grid=True)["contract_sha256"],
        )
        self.assertNotEqual(same["contract_sha256"], other["contract_sha256"])
        self.assertEqual(same["problem_ids_sha256"], other["problem_ids_sha256"])

    def test_absent_contract_is_recorded_as_unfixed_grid(self) -> None:
        summary = summarize_validation_contract(
            None,
            fixed_grid=False,
            seed=3,
            note="phase1 parity pending",
        )
        self.assertFalse(summary["fixed_grid"])
        self.assertEqual(summary["problem_count"], 0)
        self.assertEqual(summary["contract_sha256"], "")
        self.assertEqual(summary["note"], "phase1 parity pending")

    def test_canonical_sha256_is_key_order_independent(self) -> None:
        self.assertEqual(
            canonical_sha256({"a": 1, "b": [1, 2]}),
            canonical_sha256({"b": [1, 2], "a": 1}),
        )


if __name__ == "__main__":
    unittest.main()
