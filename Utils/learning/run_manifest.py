"""학습/평가 run의 identity·코드·환경·계약을 시작 시점에 고정 기록한다.

여러 사람이 각자 머신에서 arm을 나눠 실행한 뒤 결과를 합쳐 비교하려면, 어떤 코드와
어떤 문제집합으로 만들어진 산출물인지 run 자신이 증명해야 한다. `summary.json`은 학습이
끝나야 써지므로 중단된 run은 감사할 수 없고, eval-only summary는 계약을 담지 않는다.
이 모듈은 첫 episode 이전에 `run_manifest.json`을 남겨 그 공백을 메운다.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


RUN_MANIFEST_SCHEMA = "fab_run_manifest_v1"
RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_MANIFEST_HISTORY_FILENAME = "run_manifest_history.jsonl"

# manifest의 실험 identity 필드. CLI에서 그대로 받아 기록한다.
RUN_MANIFEST_IDENTITY_FIELDS = (
    "experiment_id",
    "arm_group",
    "arm_label",
    "owner",
    "run_note",
)

# 같은 output 디렉터리로 다시 실행할 때 달라지면 안 되는 키. 하나라도 다르면 서로 다른
# 실험 산출물이 한 디렉터리에 섞이므로 fallback 없이 실패시킨다.
_RUN_MANIFEST_LOCKED_IDENTITY = ("experiment_id", "arm_group", "arm_label")

# code fingerprint 계산 대상. 학습 결과에 영향을 주는 소스만 포함한다.
_CODE_FINGERPRINT_ROOTS = (
    "main.py",
    "Phase1",
    "Phase2",
    "Utils",
    "Environment",
    "Train",
)

_DATA_PROFILE_RELPATH = "Utils/data/multi_series_generation_profile.json"


def repo_root() -> Path:
    """이 파일 위치에서 저장소 루트를 구한다(Utils/learning/x.py -> repo root)."""

    return Path(__file__).resolve().parents[2]


def canonical_sha256(payload: object) -> str:
    """계약 비교용 canonical JSON SHA-256."""

    encoded = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cli_manifest_fields(
    args: object,
    *,
    command: str,
    phase: str,
) -> dict:
    """argparse Namespace에서 manifest의 CLI 절반을 만든다."""

    # argparse의 `set_defaults(func=...)` 같은 callable은 repr에 메모리 주소가 들어가
    # 매 실행마다 달라진다. 비교 단계에서 가짜 arm 차이로 잡히므로 제외한다.
    resolved_args = _json_safe(
        {key: value for key, value in vars(args).items() if not callable(value)}
    )
    fields = {
        "phase": phase,
        "command": command,
        "cli_argv": list(sys.argv),
        "resolved_args": resolved_args,
    }
    for key in RUN_MANIFEST_IDENTITY_FIELDS:
        value = getattr(args, key, None)
        fields[key] = "" if value is None else str(value)
    if not fields["owner"]:
        fields["owner"] = os.environ.get("USER", "")
    return fields


def summarize_validation_contract(
    contract: Mapping[str, object] | None,
    *,
    fixed_grid: bool,
    seed: int | None = None,
    note: str = "",
) -> dict:
    """validation 문제집합을 manifest용 지문으로 요약한다."""

    if contract is None:
        return {
            "fixed_grid": bool(fixed_grid),
            "schema": "",
            "problem_count": 0,
            "contract_sha256": "",
            "problem_ids_sha256": "",
            "block_sizes": [],
            "type_count": 0,
            "seed": seed,
            "note": note,
        }
    problems = list(contract.get("problems") or ())
    problem_ids = sorted(str(problem.get("problem_id", "")) for problem in problems)
    block_sizes = sorted({int(problem["block_count"]) for problem in problems if "block_count" in problem})
    types = {int(problem["distribution_type"]) for problem in problems if "distribution_type" in problem}
    return {
        "fixed_grid": bool(fixed_grid),
        "schema": str(contract.get("schema", "")),
        "problem_count": len(problems),
        "contract_sha256": canonical_sha256(contract),
        "problem_ids_sha256": canonical_sha256(problem_ids),
        "block_sizes": block_sizes,
        "type_count": len(types),
        "seed": seed,
        "note": note,
    }


def build_run_manifest(
    *,
    output_dir: str | Path,
    cli_fields: Mapping[str, object],
    run_spec: Mapping[str, object] | None,
    run_spec_source: str,
    validation: Mapping[str, object],
    extra: Mapping[str, object] | None = None,
) -> dict:
    """기록할 manifest dict를 만든다(파일 쓰기는 하지 않는다)."""

    missing = [key for key in ("phase", "command") if not cli_fields.get(key)]
    if missing:
        print(
            "[ERROR][Utils.learning.run_manifest.build_run_manifest] "
            f"cause=missing_cli_fields keys={missing}"
        )
        raise RuntimeError("run manifest requires phase and command in cli_fields")
    if not isinstance(validation, Mapping):
        print(
            "[ERROR][Utils.learning.run_manifest.build_run_manifest] "
            f"cause=invalid_validation_section type={type(validation).__name__}"
        )
        raise RuntimeError("run manifest validation section must be a mapping")
    root = repo_root()
    identity = {key: str(cli_fields.get(key, "") or "") for key in RUN_MANIFEST_IDENTITY_FIELDS}
    identity["host"] = platform.node()
    identity["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {
        "schema": RUN_MANIFEST_SCHEMA,
        "identity": identity,
        "invocation": {
            "phase": str(cli_fields["phase"]),
            "command": str(cli_fields["command"]),
            "cli_argv": list(cli_fields.get("cli_argv") or ()),
            "resolved_args": _json_safe(cli_fields.get("resolved_args") or {}),
            "output_dir": str(output_dir),
        },
        "code": {
            "git": git_state(root),
            "code_fingerprint": code_fingerprint(root),
        },
        "env": environment_state(str((extra or {}).get("device", ""))),
        "data": data_profile_state(root),
        "contract": {
            "run_spec": _json_safe(run_spec) if run_spec is not None else None,
            "run_spec_source": run_spec_source,
        },
        "validation": _json_safe(validation),
        "extra": _json_safe(extra or {}),
    }
    return manifest


def write_run_manifest(
    output_dir: str | Path,
    *,
    cli_fields: Mapping[str, object],
    run_spec: Mapping[str, object] | None,
    run_spec_source: str,
    validation: Mapping[str, object],
    extra: Mapping[str, object] | None = None,
) -> Path:
    """`<output_dir>/run_manifest.json`을 쓴다. 이전 manifest는 history로 넘긴다."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / RUN_MANIFEST_FILENAME
    manifest = build_run_manifest(
        output_dir=output_path,
        cli_fields=cli_fields,
        run_spec=run_spec,
        run_spec_source=run_spec_source,
        validation=validation,
        extra=extra,
    )
    if manifest_path.is_file():
        previous = _read_manifest(manifest_path)
        _require_compatible_manifest(previous, manifest, manifest_path)
        _append_manifest_history(output_path, previous)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    validation_main = manifest["validation"].get("main") or {}
    print(
        "[CHECK][Utils.learning.run_manifest.write_run_manifest] "
        f"manifest={manifest_path} phase={manifest['invocation']['phase']} "
        f"experiment_id={manifest['identity']['experiment_id']} "
        f"arm_label={manifest['identity']['arm_label']} "
        f"code_fingerprint={manifest['code']['code_fingerprint']['digest'][:12]} "
        f"git_dirty={str(manifest['code']['git'].get('dirty', '')).lower()} "
        f"validation_contract_sha256={str(validation_main.get('contract_sha256', ''))[:12]} "
        f"validation_problem_count={validation_main.get('problem_count', 0)}"
    )
    return manifest_path


def read_run_manifest(run_dir: str | Path) -> dict:
    """run 디렉터리의 manifest를 읽는다. 없거나 schema가 다르면 실패한다."""

    manifest_path = Path(run_dir) / RUN_MANIFEST_FILENAME
    if not manifest_path.is_file():
        print(
            "[ERROR][Utils.learning.run_manifest.read_run_manifest] "
            f"cause=missing_run_manifest path={manifest_path}"
        )
        raise RuntimeError(f"run manifest not found: {manifest_path}")
    manifest = _read_manifest(manifest_path)
    schema = str(manifest.get("schema", ""))
    if schema != RUN_MANIFEST_SCHEMA:
        print(
            "[ERROR][Utils.learning.run_manifest.read_run_manifest] "
            f"cause=run_manifest_schema_mismatch path={manifest_path} "
            f"expected={RUN_MANIFEST_SCHEMA} actual={schema}"
        )
        raise RuntimeError("unsupported run manifest schema")
    return manifest


def code_fingerprint(root: str | Path | None = None) -> dict:
    """학습 결과에 영향을 주는 소스 파일들의 결정적 지문."""

    base = Path(root) if root is not None else repo_root()
    entries: list[tuple[str, str]] = []
    for relative in _CODE_FINGERPRINT_ROOTS:
        target = base / relative
        if target.is_file():
            entries.append((relative, _file_sha256(target)))
            continue
        if not target.is_dir():
            continue
        for path in sorted(target.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            entries.append((path.relative_to(base).as_posix(), _file_sha256(path)))
    entries.sort()
    digest = hashlib.sha256(
        "\n".join(f"{name}:{file_hash}" for name, file_hash in entries).encode("utf-8")
    ).hexdigest()
    return {
        "algorithm": "sha256",
        "roots": list(_CODE_FINGERPRINT_ROOTS),
        "file_count": len(entries),
        "digest": digest,
    }


def git_state(root: str | Path | None = None) -> dict:
    """git commit/branch/dirty 상태. git 사용 불가도 숨기지 않고 기록한다."""

    base = Path(root) if root is not None else repo_root()
    commit = _git_output(base, ["rev-parse", "HEAD"])
    if commit is None:
        print(
            "[CHECK][Utils.learning.run_manifest.git_state] "
            f"git_available=false root={base}"
        )
        return {"available": False, "commit": "", "branch": "", "dirty": None, "diff_sha256": ""}
    status = _git_output(base, ["status", "--porcelain"]) or ""
    diff = _git_output(base, ["diff", "HEAD"]) or ""
    return {
        "available": True,
        "commit": commit,
        "branch": _git_output(base, ["rev-parse", "--abbrev-ref", "HEAD"]) or "",
        "describe": _git_output(base, ["describe", "--tags", "--always", "--dirty"]) or "",
        "dirty": bool(status.strip()),
        "status_line_count": len([line for line in status.splitlines() if line.strip()]),
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def environment_state(device: str = "") -> dict:
    """실행 환경. torch/numpy import 실패도 값으로 남긴다."""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device,
        "cpu_count": os.cpu_count(),
        "torch": _module_version("torch"),
        "numpy": _module_version("numpy"),
    }


def data_profile_state(root: str | Path | None = None) -> dict:
    """학습 데이터의 단일 출처인 고정 generation profile의 지문."""

    base = Path(root) if root is not None else repo_root()
    profile_path = base / _DATA_PROFILE_RELPATH
    if not profile_path.is_file():
        print(
            "[ERROR][Utils.learning.run_manifest.data_profile_state] "
            f"cause=missing_generation_profile path={profile_path}"
        )
        raise RuntimeError(f"generation profile not found: {profile_path}")
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(
            "[ERROR][Utils.learning.run_manifest.data_profile_state] "
            f"cause=unreadable_generation_profile path={profile_path}"
        )
        raise RuntimeError("generation profile could not be read") from exc
    sources = profile.get("sources")
    if not isinstance(sources, Mapping):
        print(
            "[ERROR][Utils.learning.run_manifest.data_profile_state] "
            f"cause=missing_profile_sources path={profile_path}"
        )
        raise RuntimeError("generation profile has no sources section")
    return {
        "profile_path": _DATA_PROFILE_RELPATH,
        "profile_sha256": _file_sha256(profile_path),
        "schema": str(profile.get("schema", "")),
        "sources": {
            str(name): {
                "path": str(source.get("path", "")),
                "sha256": str(source.get("sha256", "")),
            }
            for name, source in sorted(sources.items())
        },
    }


def _require_compatible_manifest(
    previous: Mapping[str, object],
    current: Mapping[str, object],
    manifest_path: Path,
) -> None:
    previous_identity = previous.get("identity") or {}
    current_identity = current.get("identity") or {}
    mismatched = [
        key
        for key in _RUN_MANIFEST_LOCKED_IDENTITY
        if str(previous_identity.get(key, "")) != str(current_identity.get(key, ""))
    ]
    previous_code = ((previous.get("code") or {}).get("code_fingerprint") or {}).get("digest", "")
    current_code = ((current.get("code") or {}).get("code_fingerprint") or {}).get("digest", "")
    if previous_code != current_code:
        mismatched.append("code_fingerprint")
    if not mismatched:
        return
    print(
        "[ERROR][Utils.learning.run_manifest._require_compatible_manifest] "
        f"cause=run_manifest_mismatch path={manifest_path} keys={mismatched} "
        f"previous_identity={ {key: previous_identity.get(key, '') for key in _RUN_MANIFEST_LOCKED_IDENTITY} } "
        f"current_identity={ {key: current_identity.get(key, '') for key in _RUN_MANIFEST_LOCKED_IDENTITY} } "
        f"previous_code_fingerprint={previous_code[:12]} current_code_fingerprint={current_code[:12]}"
    )
    raise RuntimeError(
        "output directory already belongs to a different experiment run; "
        "use a new --output-dir instead of mixing runs"
    )


def _append_manifest_history(output_path: Path, previous: Mapping[str, object]) -> None:
    history_path = output_path / RUN_MANIFEST_HISTORY_FILENAME
    entry = dict(previous)
    entry["previous_manifest_sha256"] = canonical_sha256(previous)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _read_manifest(manifest_path: Path) -> dict:
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(
            "[ERROR][Utils.learning.run_manifest._read_manifest] "
            f"cause=unreadable_run_manifest path={manifest_path}"
        )
        raise RuntimeError(f"run manifest could not be read: {manifest_path}") from exc


def _git_output(root: Path, arguments: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(
            "[CHECK][Utils.learning.run_manifest._git_output] "
            f"git_call_failed args={list(arguments)} cause={exc}"
        )
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _module_version(name: str) -> str:
    try:
        module = __import__(name)
    except ImportError as exc:
        return f"import_failed:{exc}"
    return str(getattr(module, "__version__", "unknown"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        if isinstance(value, (set, frozenset)):
            items = sorted(items, key=str)
        return [_json_safe(item) for item in items]
    if isinstance(value, Path):
        return str(value)
    return str(value)
