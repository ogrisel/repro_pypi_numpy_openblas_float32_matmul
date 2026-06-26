#!/usr/bin/env python
"""Run the reproducer in every pixi environment and write one JSON file per env."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

# Environments declared in pixi.toml; platform filters match workspace targets.
ALL_ENVIRONMENTS = [
    "pypi",
    "openblas-pthreads",
    "openblas-openmp",
    "newaccelerate",
    "mkl",
]

PLATFORM_FILTERS = {
    "newaccelerate": {"Darwin"},
    "mkl": {"Linux", "Darwin", "Windows"},
}


def compatible_environments() -> list[str]:
    system = platform.system()
    machine = platform.machine().lower()
    envs = []
    for name in ALL_ENVIRONMENTS:
        allowed = PLATFORM_FILTERS.get(name)
        if allowed is not None and system not in allowed:
            continue
        # MKL environment is not defined for osx-arm64 in pixi.toml.
        if name == "mkl" and system == "Darwin" and machine in {"arm64", "aarch64"}:
            continue
        envs.append(name)
    return envs


def run_environment(env: str) -> dict:
    command = ["pixi", "run", "-e", env, "python", "repro.py", "--json"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    record: dict = {
        "environment": env,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "repro": None,
        "error": None,
    }
    if completed.stdout.strip():
        try:
            record["repro"] = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            record["error"] = f"invalid JSON from repro.py: {exc}"
    elif completed.returncode != 0 and completed.stderr.strip():
        record["error"] = completed.stderr.strip()
    return record


def platform_slug() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64"}:
        machine = "x86_64"
    elif machine in {"aarch64", "arm64"}:
        machine = "arm64"
    aliases = {"darwin": "macos", "windows": "win"}
    system = aliases.get(system, system.replace(" ", "-"))
    return f"{system}-{machine}"


def result_filename(env: str) -> str:
    return f"{env}-{platform_slug()}.json"


def write_environment_result(env: str, run: dict) -> Path:
    output_path = RESULTS_DIR / result_filename(env)
    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "host_platform": platform.platform(),
        "platform_slug": platform_slug(),
        "environment": env,
        "exit_code": run["exit_code"],
        "error": run["error"],
        "repro": run["repro"],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    environments = compatible_environments()
    runs = []
    for env in environments:
        run = run_environment(env)
        runs.append(run)
        output_path = write_environment_result(env, run)
        print(f"Wrote {output_path}")

    for run in runs:
        repro = run["repro"] or {}
        status = "OK" if not repro.get("reproduces") else "FAIL"
        extra = ""
        checks = repro.get("checks") or {}
        matmul = checks.get("matmul") or {}
        if repro.get("reproduces"):
            extra = f" (matmul nan_count={matmul.get('nan_count')})"
        elif run.get("error"):
            extra = f" ({run['error']})"
        print(f"  {run['environment']}: {status}{extra}")

    failing = [
        run["environment"]
        for run in runs
        if run.get("repro") and run["repro"].get("reproduces")
    ]
    if failing:
        print(f"Reproduces in: {', '.join(failing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
