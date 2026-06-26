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
    "pypi-openblas",
    "pypi-accelerate",
    "openblas-pthreads",
    "openblas-openmp",
    "newaccelerate",
    "mkl",
]

# On macOS arm64, openblas conda envs are collected for both declared platforms.
MACOS_ARM64_OPENBLAS_PLATFORMS = [
    "osx-arm64-macos-11-0",
    "osx-arm64-macos-15-5",
]


def _try_parse_json(stdout: str) -> dict | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Some runtimes prepend/append logs to stdout. Try to recover a JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _compose_error(stdout: str, stderr: str) -> str | None:
    parts = []
    if stdout.strip():
        parts.append(f"stdout: {stdout.strip()}")
    if stderr.strip():
        parts.append(f"stderr: {stderr.strip()}")
    return "\n".join(parts) if parts else None


PLATFORM_FILTERS = {
    "pypi": {"Linux", "Windows"},
    "pypi-openblas": {"Darwin"},
    "pypi-accelerate": {"Darwin"},
    "newaccelerate": {"Darwin"},
    "mkl": {"Linux", "Darwin", "Windows"},
}


def compatible_runs() -> list[tuple[str, str | None]]:
    """Return (environment, optional pixi --platform) runs for this host."""
    system = platform.system()
    machine = platform.machine().lower()
    runs: list[tuple[str, str | None]] = []
    for name in ALL_ENVIRONMENTS:
        allowed = PLATFORM_FILTERS.get(name)
        if allowed is not None and system not in allowed:
            continue
        # MKL environment is not defined for osx-arm64 in pixi.toml.
        if name == "mkl" and system == "Darwin" and machine in {"arm64", "aarch64"}:
            continue
        if name in {"pypi-openblas", "pypi-accelerate"} and system == "Darwin":
            if machine not in {"arm64", "aarch64"}:
                continue
        if (
            name in {"openblas-pthreads", "openblas-openmp"}
            and system == "Darwin"
            and machine in {"arm64", "aarch64"}
        ):
            for pixi_platform in MACOS_ARM64_OPENBLAS_PLATFORMS:
                runs.append((name, pixi_platform))
            continue
        runs.append((name, None))
    return runs


def run_environment(env: str, pixi_platform: str | None = None) -> dict:
    command = ["pixi", "run", "-e", env]
    if pixi_platform is not None:
        command.extend(["--platform", pixi_platform])
    command.extend(["python", "repro.py", "--json"])
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    record: dict = {
        "environment": env,
        "pixi_platform": pixi_platform,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "repro": None,
        "error": None,
    }
    record["repro"] = _try_parse_json(completed.stdout)

    if completed.returncode != 0 and record["repro"] is None:
        record["error"] = _compose_error(completed.stdout, completed.stderr)
    elif completed.returncode == 0 and completed.stdout.strip() and record["repro"] is None:
        # Keep this explicit for successful commands that emit malformed JSON.
        record["error"] = "invalid JSON from repro.py output"

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


def result_filename(env: str, pixi_platform: str | None = None) -> str:
    if pixi_platform is not None:
        return f"{env}-{pixi_platform}.json"
    return f"{env}-{platform_slug()}.json"


def write_environment_result(env: str, run: dict) -> Path:
    output_path = RESULTS_DIR / result_filename(env, run.get("pixi_platform"))
    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "host_platform": platform.platform(),
        "platform_slug": platform_slug(),
        "pixi_platform": run.get("pixi_platform"),
        "environment": env,
        "exit_code": run["exit_code"],
        "error": run["error"],
        "stdout": run["stdout"],
        "stderr": run["stderr"],
        "repro": run["repro"],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    environments = compatible_runs()
    runs = []
    for env, pixi_platform in environments:
        run = run_environment(env, pixi_platform)
        runs.append(run)
        output_path = write_environment_result(env, run)
        print(f"Wrote {output_path}")

    for run in runs:
        repro = run["repro"] or {}
        if run.get("error"):
            status = "ERROR"
        elif repro.get("reproduces"):
            status = "FAIL"
        elif run.get("repro") is None:
            status = "ERROR"
        else:
            status = "OK"
        extra = ""
        checks = repro.get("checks") or {}
        matmul = checks.get("matmul") or {}
        label = run["environment"]
        if run.get("pixi_platform"):
            label = f"{label} ({run['pixi_platform']})"
        if repro.get("reproduces"):
            extra = f" (matmul nan_count={matmul.get('nan_count')})"
        elif run.get("error"):
            extra = f" ({run['error']})"
        print(f"  {label}: {status}{extra}")

    failing = [
        run["environment"]
        for run in runs
        if run.get("repro") and run["repro"].get("reproduces")
    ]
    errored = [run["environment"] for run in runs if run.get("error")]

    if errored:
        print(f"Execution errors in: {', '.join(errored)}")
        return 2
    if failing:
        print(f"Reproduces in: {', '.join(failing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
