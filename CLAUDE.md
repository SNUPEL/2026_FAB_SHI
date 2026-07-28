# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read these first

- **`AGENTS.md`** — the authoritative working rules (Korean). It defines the mandatory
  no-silent-fallback policy, error-output format, data contracts, and the required
  verification loop. Treat it as binding, not advisory.
- **`현재과제_진행체크리스트.md`** — the single master checklist. Current state and next
  tasks live here only; update it before touching code. Other `*.md` files are archive/reference.
- **`README.md`** — problem definition (Phase 1 / Phase 2), data contracts, and run commands.

## Environment & commands

This is a **Windows / PowerShell** repo with a conda-style env. The docs write `python3`, but
locally use **`python`**. A `.venv` exists at the repo root.

```powershell
# Compile check (fast smoke)
python -m py_compile main.py Agent/*.py Phase1/*.py Phase2/*.py Environment/*.py Environment/constraints/*.py Utils/data/*.py Utils/learning/*.py Utils/phase1/*.py Utils/reporting/*.py Train/network/*.py scripts/*.py

# Full test suite (unittest, not pytest)
python -m unittest discover -s tests -p 'test_*.py'

# Single test module / case
python -m unittest tests.test_phase2_feedback
python -m unittest tests.test_phase2_feedback.SomeTestCase.test_method

# NP100 regression (the canonical "did I break it" gate — must stay green)
python main.py show-config      --config config_np_100.yaml
python main.py factory-summary  --config config_np_100.yaml
python main.py simulate         --config config_np_100.yaml --heuristic spt
python main.py simulate         --config config_np_100.yaml --heuristic load_balance
python main.py factory-replay   --config config_np_100.yaml
```

Regression pass criteria: SPT and load_balance each `scheduled=100, unscheduled=0, event=800`;
actual-replay identity error `=0`. `holidays` is required for `phase1-plan-multi-series`
(Korean legal/substitute holidays) and must not fall back to a default calendar.

`main.py` is a single ~227KB argparse CLI (`main()` at the bottom, `add_subparsers`). Every line
carries an auto-generated Korean `# LINE-BY-LINE:` comment — these are machine-written noise, not
design intent; do not treat them as authoritative and do not replicate the style in new code.

Key subcommands: `simulate`, `playback`, `actual-replay` / `factory-replay`, `factory-summary`,
`generate-phase1-blocks`, `phase1-train-pair-self-labeling`,
`phase2-train-batch-machine-self-labeling`, `phase2-run-full-workflow`, `phase1-plan-multi-series`,
`build-scenario`, `analyze-tact`, `analyze-tact-gap`.

## Architecture

Two-phase scheduler for a shipbuilding steel-cutting shop, over four block series **NP/FN/FL/NC**.
The only supported ("public") learning path is **MIXED**; NP-only / imitation / 3-Bay legacy paths
are gone.

- **Phase 1** (`Phase1/`, `Utils/phase1/`) — assign each **block-series** `(PROJ_NO, GYEL, BLK_NO)`
  to a **Bay** (`22/23/24/25/trans`). A physical block with different series becomes distinct
  block-series work; all W/O of one block-series share a Bay. At train time one parent episode is
  split into two **independent resource-pool subproblems**: `NP_NC` → Bays 22/23/24, `FN_FL` →
  Bays 25/trans. One shared pointer policy gets two sequential CE updates (never a merged teacher
  score). Pair/env feature schema is **`20+8`**, checkpoint scope `resource_pool_subproblems_v1`.
- **Phase 2** (`Phase2/`) — a single `SELECT_WO` Set-Pointer policy fills the **open batch of a
  machine the environment has already chosen** (idle machine at global time, ascending `machine_id`).
  Batch = 1–3 W/O, `LTH` sum ≤ 55,000, processing time = max `TACT_TIME`. There is no
  `SELECT_MACHINE` action. RunSpec is `phase2_run_spec_v4_event_clock`.
- **`Environment/`** — a **custom event-driven DES** (`CuttingSimulation`, `simulation.py`);
  **not SimPy**. Generated schedules and actual replay share one event schema. Constraints live in
  `Environment/constraints/` as small `ConstraintContext -> ConstraintResult` functions registered
  in `registry.py` (`RULES` + `RULE_CATEGORIES`), toggled via config `hard_enabled` / `soft_enabled`.
  `CommonHierarchicalEnvironment` (`hierarchical.py`) holds shared planning state + machine clock
  for both phases.
- **`Utils/data/`** — strict loaders and the synthetic-data generators. The MIXED generator
  (`multi_series_formula_data_generator.py`) reads only the frozen
  `multi_series_generation_profile.json` (source SHA256 + fitted coefficients); it never re-fits the
  source Excel. Regenerate the profile deliberately with
  `scripts/build_multi_series_generation_profile.py` (output must be byte-for-byte identical).
  `report_formula_data_generator.py` is the **NP-only** generator that runs the PDF report formulas
  directly (used by main.py, the MIXED generator, and `Utils/phase1/phase1_episode_dataset.py`).
- **`데이터분석/`** (Korean "data analysis") — despite the name, this is an **imported package**, not
  scratch: `Utils/data/multi_series_formula_data_generator.py` imports `ShipyardGenerator` from
  `데이터분석/shipyard_data_generator.py`. That file has two faces: a standalone CLI generator
  (functions `generate_blocks`/`generate_wos`/`generate`, coefficients hardcoded from
  `블록_산출식_정리.md` / `W_O_산출식_정리.md`; `--report` shows block=MAX/Σ divergence) **and** the
  `ShipyardGenerator` adapter that wires FN/FL/NC generation into MIXED training. The adapter honors
  the pipeline's `generate(n_blocks, seed, wo_counts, block_seeds)` contract, emits a `BLK_ID` group
  key + Case-6 `TACT_TIME`, and serializes its coefficients into the fixed profile
  (`shipyard_formula_generation_profile_v2`). NP is **not** served by this adapter — it stays on
  `report_formula_data_generator.py`. The fitting/generation source is `input/260724_절단*_None.xlsx`
  (`DEFAULT_MULTI_SERIES_WO_SOURCE` / `DEFAULT_MULTI_SERIES_BLOCK_SOURCE`), which has block
  `MARK_LTH = sum` (not max) and **per-W/O `STL_QTY = 1` for all series** — so block `STL_QTY == WO_QTY`
  everywhere (the code still sums the `STL_QTY` column, not the row count). The `fit_blocks_params.py` / `fit_wo_params.py`
  fitters here produce `block_params.json` (`block_formula_params_v3`) and `wo_params.json`
  (`wo_formula_params_v3`) plus the `bth*.py` fitters and `heatmap_*` dirs — those JSONs are still
  **analysis outputs, not consumed by the training path**. Tests: `tests/test_shipyard_data_generator.py`.
  Regenerate the profile deliberately with `scripts/build_multi_series_generation_profile.py` (output
  must stay byte-for-byte reproducible).
- **`Train/network/mlp.py`** — MLP builder shared by both phase policies. **`Agent/heuristics.py`**
  — NP100 DES baseline heuristics.

### Factory topologies — do not conflate

- **MIXED Phase 1/2 planning** uses **15** PLS/PLP machines: Bay22 `PLS21-24`, Bay23 `PLS31-33`,
  Bay24 `PLS41-44`, Bay25 `PLS51-52`, trans `PLP01-02`. Machine identity is defined once in
  `Utils/data/multi_series_cutting_data.py` (`MIXED_PLANNING_MACHINE_IDS_BY_BAY`); capacities are
  derived from it, never duplicated as separate numbers.
- **`config_np_100.yaml` / `config_np_full.yaml` DES/replay** regress the historical NP actual
  scope and use a **13**-machine factory. Keep these separate from the 15-machine planning topology.
- Historical `EQP_1..16` map to fixed PLS/PLP identities, **except `EQP_3`**, an actual-only NC/trans
  machine: preserve it in replay identity but exclude it from NC planning candidates.

## Conventions that bite

- **No silent fallback** (see AGENTS.md §4–5): missing columns, date/parse failures, TACT/mapping
  failures, and hard-constraint evaluation errors must `print` a diagnostic and `raise` — never
  substitute mean/0/first-machine/pass. Broad `except` must re-raise via `raise RuntimeError(...) from exc`.
- `TACT_TIME` (minutes, arc-cutting time) is the planning process time. Do **not** substitute actual
  start→end elapsed time; elapsed is for replay identity / audit only.
- `CUT_BAY` (cutting Bay) and `downstream_bay` are different fields — never mix them.
- NP 장척 (long-piece) is judged by the **sum** of `CUT_LTH` over one `PROJ_NO+GYEL+BLK_NO`'s W/O
  being `>= 1000` (not any single W/O max); it plus `BTH > 4500` and CNT blocks are Bay 22/23 hard masks.
- Regression baseline is **`config_np_100.yaml`** (builds 100 scenarios in memory from
  `input/절단03~04_NP물량_마스킹_WO_수정_260618.xlsx`). Do not develop or validate against full data first.
- Generated artifacts go under `output/generated/`; `input/` is source-only, not a scratch bin.
- Old `joint_five_bay_*` checkpoints (19-dim pair, single teacher) are incompatible and rejected.
