#!/usr/bin/env python3
"""Create a clean workspace and reproduce the complete frozen CPU analysis chain."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


RELEASE_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = Path("data/processed/all_molglue_dc50_qc_train_test_standardized_context.csv")
SEED = "260531"


def command_text(command: Iterable[object]) -> str:
    return " ".join(str(item) for item in command)


def write_state(root: Path, state: dict[str, object]) -> None:
    state["updated_at_epoch"] = time.time()
    (root / "cpu_reproduction_state_v2.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def run_command(
    root: Path,
    name: str,
    command: list[object],
    *,
    environment: dict[str, str] | None = None,
) -> None:
    logs = root / "runtime_logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{name}.log"
    env = os.environ.copy()
    if environment:
        env.update(environment)
    print(f"START {name}", flush=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"COMMAND: {command_text(command)}\n")
        handle.flush()
        result = subprocess.run(
            [str(item) for item in command],
            cwd=root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}; see {log_path}")
    print(f"PASS  {name} ({time.time() - started:.1f} s)", flush=True)


def run_parallel_null(root: Path, python: str, resume: bool) -> None:
    logs = root / "runtime_logs"
    logs.mkdir(parents=True, exist_ok=True)
    base = [python, "analysis_code"]
    jobs: list[tuple[str, list[str]]] = [
        (
            "null_primary_001_025",
            [
                python,
                "analysis_code/run_post_hoc_null_applicability_censoring_v3.py",
                "--output-dir",
                "reports/post_hoc_null_applicability_censoring_v3",
                "--n-permutations",
                "25",
                "--n-estimators",
                "600",
                "--n-jobs",
                "4",
                "--bootstrap-replicates",
                "10000",
                "--max-repeats",
                "5",
                "--seed",
                SEED,
            ],
        )
    ]
    for start, end in ((26, 50), (51, 75), (76, 100)):
        jobs.append(
            (
                f"null_shard_{start:03d}_{end:03d}",
                [
                    python,
                    "analysis_code/run_post_hoc_permutation_shard_v3.py",
                    "--start-id",
                    str(start),
                    "--end-id",
                    str(end),
                    "--output-dir",
                    f"reports/post_hoc_null_permutation_shard_{start:03d}_{end:03d}_v3",
                    "--n-estimators",
                    "600",
                    "--n-jobs",
                    "4",
                    "--max-repeats",
                    "5",
                    "--seed",
                    SEED,
                ],
            )
        )
    if resume:
        for _, command in jobs:
            command.append("--resume")
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp/mplconfig"
    processes: list[tuple[str, subprocess.Popen[bytes], object, Path]] = []
    print("START null_permutation_workers (4 isolated CPU workers)", flush=True)
    for name, command in jobs:
        log_path = logs / f"{name}.log"
        handle = log_path.open("wb")
        handle.write((f"COMMAND: {command_text(command)}\n").encode("utf-8"))
        handle.flush()
        process = subprocess.Popen(command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((name, process, handle, log_path))
    failures = []
    try:
        for name, process, handle, log_path in processes:
            returncode = process.wait()
            handle.close()
            if returncode:
                failures.append(f"{name} exit {returncode}; see {log_path}")
    finally:
        for _, process, handle, _ in processes:
            if process.poll() is None:
                process.terminate()
            if not handle.closed:
                handle.close()
    if failures:
        raise RuntimeError("; ".join(failures))
    print("PASS  null_permutation_workers", flush=True)
    shard_dirs = [
        f"reports/post_hoc_null_permutation_shard_{start:03d}_{end:03d}_v3"
        for start, end in ((26, 50), (51, 75), (76, 100))
    ]
    merge = [
        python,
        "analysis_code/merge_post_hoc_permutation_shards_v3.py",
        "--primary-dir",
        "reports/post_hoc_null_applicability_censoring_v3",
    ]
    for directory in shard_dirs:
        merge.extend(["--shard-dir", directory])
    merge.extend(["--n-permutations", "100"])
    run_command(root, "null_merge_100", merge)
    run_command(
        root,
        "null_finalize_100",
        [
            python,
            "analysis_code/run_post_hoc_null_applicability_censoring_v3.py",
            "--output-dir",
            "reports/post_hoc_null_applicability_censoring_v3",
            "--n-permutations",
            "100",
            "--n-estimators",
            "600",
            "--n-jobs",
            "4",
            "--bootstrap-replicates",
            "10000",
            "--max-repeats",
            "5",
            "--seed",
            SEED,
            "--resume",
        ],
        environment={"MPLCONFIGDIR": "/tmp/mplconfig"},
    )


def public_copy(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"clean work directory already exists: {target}")

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in {".git", "__pycache__", ".pytest_cache", "reports", "runtime", "runtime_logs"}}
        return ignored

    shutil.copytree(source, target, ignore=ignore)


def seed_source_cache(work_root: Path, source_cache: Path) -> None:
    manifest = json.loads(
        (work_root / "data_builder/config/source_manifest.json").read_text(encoding="utf-8")
    )
    required: set[str] = set()
    for source in manifest["sources"]:
        for item in source["files"]:
            if item.get("required"):
                required.add(item["filename"])
                if item.get("extracted_filename"):
                    required.add(item["extracted_filename"])
    destination = work_root / "data/raw"
    destination.mkdir(parents=True, exist_ok=True)
    missing = []
    for filename in sorted(required):
        source = source_cache / filename
        if not source.is_file():
            missing.append(filename)
        else:
            shutil.copy2(source, destination / filename)
    if missing:
        raise FileNotFoundError(f"source cache is missing required artifacts: {missing}")


def scientific_environment_check(root: Path, name: str, python: str, expected: dict[str, str]) -> None:
    expression = (
        "import json,sys; "
        "mods={n:__import__(n) for n in " + repr(list(expected)) + "}; "
        "print(json.dumps({'python':'.'.join(map(str,sys.version_info[:3])),"
        "'versions':{n:getattr(m,'__version__','?') for n,m in mods.items()}}))"
    )
    log = subprocess.run([python, "-c", expression], cwd=root, check=True, capture_output=True, text=True)
    payload = json.loads(log.stdout.strip().splitlines()[-1])
    mismatches = {
        module: (payload["versions"].get(module), version)
        for module, version in expected.items()
        if payload["versions"].get(module) != version
    }
    if mismatches:
        raise RuntimeError(f"{name} environment mismatch: {mismatches}")
    print(f"PASS  {name}_environment: Python {payload['python']} {payload['versions']}", flush=True)


def stage_commands(analysis_python: str) -> list[tuple[str, list[object], dict[str, str] | None]]:
    py = analysis_python
    data = str(DATA_FILE)
    common = {"MPLCONFIGDIR": "/tmp/mplconfig"}
    return [
        ("confirmatory_core", [py, "analysis_code/run_confirmatory_cpu_v1.py", "--protocols", "scaffold", "compound", "--data-file", data, "--output-dir", "reports/confirmatory_cpu_v1", "--outer-folds", "5", "--outer-repeats", "5", "--inner-folds", "4", "--n-estimators", "600", "--n-jobs", "10", "--bootstrap-replicates", "10000", "--seed", SEED, "--overwrite"], common),
        ("matched_context", [py, "analysis_code/run_context_et_matched_ablation_cpu_v1.py", "--protocols", "scaffold", "compound", "--core-results-dir", "reports/confirmatory_cpu_v1", "--output-dir", "reports/context_et_matched_ablation_cpu_v1", "--n-jobs", "10", "--overwrite"], common),
        ("confirmatory_ood", [py, "analysis_code/run_confirmatory_ood_cpu_v1.py", "--protocols", "source_ood", "target_ood", "--data-file", data, "--output-dir", "reports/confirmatory_ood_cpu_v1", "--inner-folds", "4", "--n-estimators", "600", "--n-jobs", "10", "--bootstrap-replicates", "10000", "--seed", SEED, "--overwrite"], common),
        ("ood_diagnostics", [py, "analysis_code/run_confirmatory_ood_posthoc_diagnostics.py", "--formal-dir", "reports/confirmatory_ood_cpu_v1", "--output-dir", "reports/confirmatory_ood_cpu_v1_posthoc", "--protocol-note", "docs/confirmatory_ood_cpu_v1_posthoc_protocol.md", "--bootstrap-replicates", "10000", "--seed", SEED, "--overwrite"], common),
        ("strict_ood", [py, "analysis_code/run_post_hoc_strict_domain_scaffold_ood_cpu_v1.py", "--protocols", "source_ood", "target_ood", "--data-file", data, "--output-dir", "reports/post_hoc_strict_domain_scaffold_ood_cpu_v1", "--inner-folds", "4", "--n-estimators", "600", "--n-jobs", "10", "--bootstrap-replicates", "10000", "--seed", SEED, "--overwrite"], common),
        ("hgb_sensitivity", [py, "analysis_code/run_post_hoc_histgradientboosting_model_family_sensitivity_v1.py", "--output-dir", "reports/post_hoc_histgradientboosting_model_family_sensitivity_v1"], common),
        ("learning_curve", [py, "analysis_code/run_post_hoc_internal_learning_curve_v1.py", "--data-file", data, "--output-dir", "reports/post_hoc_internal_learning_curve_v1", "--n-estimators", "600", "--n-jobs", "10", "--bootstrap-replicates", "10000", "--seed", SEED, "--overwrite"], common),
        ("context_weight", [py, "analysis_code/run_post_hoc_context_weight_sensitivity_v2.py", "--output-dir", "reports/post_hoc_context_weight_sensitivity_v2", "--n-estimators", "600", "--n-jobs", "10", "--inner-folds", "4", "--bootstrap-replicates", "10000", "--seed", SEED, "--overwrite"], common),
        ("scaffold_source", [py, "analysis_code/run_post_hoc_scaffold_source_sensitivity_v3.py", "--mode", "both", "--data-file", data, "--output-dir", "reports/post_hoc_scaffold_source_sensitivity_v3", "--identity-archive-dir", "reports/confirmatory_ood_cpu_v1", "--outer-folds", "5", "--outer-repeats", "5", "--inner-folds", "4", "--n-estimators", "600", "--n-jobs", "4", "--bootstrap-replicates", "10000", "--seed", SEED, "--overwrite"], {"MPLCONFIGDIR": "/tmp/mplconfig", "OMP_NUM_THREADS": "24", "OPENBLAS_NUM_THREADS": "24"}),
    ]


def render_public_figures(root: Path, analysis_python: str) -> None:
    source = root / "evidence/source_data"
    target = root / "reports/publication_validation_v1"
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)
    common = {"MPLCONFIGDIR": "/tmp/mplconfig"}
    run_command(
        root,
        "render_figures_1_4",
        [
            analysis_python,
            "analysis_code/plot_publication_main_figures_v2.py",
            "--source-dir",
            "reports/publication_validation_v1",
            "--output-dir",
            "reports/publication_figures_v1",
        ],
        environment=common,
    )
    run_command(
        root,
        "render_figures_5_s1_s3",
        [
            analysis_python,
            "analysis_code/plot_computational_extension_figures_v1.py",
            "--source-dir",
            "reports/publication_validation_v1",
            "--existing-figure-dir",
            "reports/publication_figures_v1",
            "--output-dir",
            "reports/publication_figures_v4",
        ],
        environment=common,
    )
    run_command(
        root,
        "figure_qa",
        [
            analysis_python,
            "analysis_code/qa_computational_extension_figures_v1.py",
            "--figure-dir",
            "reports/publication_figures_v4",
            "--source-dir",
            "reports/publication_validation_v1",
        ],
        environment=common,
    )


def run_all(args: argparse.Namespace) -> int:
    work_root = args.work_dir.resolve()
    if not args.resume:
        public_copy(RELEASE_ROOT, work_root)
    elif not work_root.is_dir():
        raise FileNotFoundError(f"resume work directory does not exist: {work_root}")
    state_path = work_root / "cpu_reproduction_state_v2.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"schema_version": "2.0", "completed": []}
    completed = set(state.get("completed", []))

    scientific_environment_check(
        work_root,
        "builder",
        args.builder_python,
        {"numpy": "1.26.4", "pandas": "2.3.3", "rdkit": "2023.09.6", "sklearn": "1.3.0"},
    )
    scientific_environment_check(
        work_root,
        "analysis",
        args.analysis_python,
        {"numpy": "2.2.6", "pandas": "2.3.3", "scipy": "1.15.3", "sklearn": "1.7.2", "rdkit": "2025.09.4", "matplotlib": "3.10.8"},
    )

    if "data_build" not in completed:
        if args.source_cache:
            seed_source_cache(work_root, args.source_cache.resolve())
            commands = ("verify-sources", "build", "verify")
        else:
            commands = ("all",)
        for command_name in commands:
            command = [
                args.builder_python,
                "data_builder/scripts/reproduce.py",
                command_name,
                "--raw-dir",
                "data/raw",
                "--output-dir",
                "data/processed",
                "--manifest",
                "data_builder/config/source_manifest.json",
                "--expected",
                "data_builder/config/expected_outputs.json",
            ]
            if command_name == "all":
                command.append("--acknowledge-third-party-notice")
            run_command(work_root, f"data_{command_name}", command)
        completed.add("data_build")
        state["completed"] = sorted(completed)
        write_state(work_root, state)

    for name, command, environment in stage_commands(args.analysis_python):
        if name in completed:
            continue
        run_command(work_root, name, command, environment=environment)
        completed.add(name)
        state["completed"] = sorted(completed)
        write_state(work_root, state)

    if "null_applicability_censoring" not in completed:
        run_parallel_null(work_root, args.analysis_python, args.resume)
        completed.add("null_applicability_censoring")
        state["completed"] = sorted(completed)
        write_state(work_root, state)

    if "exact_cpu_acceptance" not in completed:
        run_command(
            work_root,
            "exact_cpu_acceptance",
            [args.builder_python, "scripts/verify_cpu_results_v2.py", "--work-root", str(work_root)],
        )
        completed.add("exact_cpu_acceptance")
        state["completed"] = sorted(completed)
        write_state(work_root, state)

    if "figure_rebuild" not in completed:
        render_public_figures(work_root, args.analysis_python)
        completed.add("figure_rebuild")
        state["completed"] = sorted(completed)
        write_state(work_root, state)

    state["status"] = "PASS"
    state["gpu_execution"] = "NOT_USED; graph-model aggregate evidence is reference-only"
    write_state(work_root, state)
    print(f"FULL CPU REPRODUCTION PASS: {work_root}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-public", help="check that the GitHub payload excludes row-level data")
    run = subparsers.add_parser("all", help="run the complete formal CPU refit in a clean work directory")
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--builder-python", default=sys.executable)
    run.add_argument("--analysis-python", default=sys.executable)
    run.add_argument("--source-cache", type=Path, help="local exact official source cache; otherwise download from owner endpoints")
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "verify-public":
        return subprocess.run([sys.executable, RELEASE_ROOT / "scripts/check_public_release_v2.py"]).returncode
    return run_all(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
