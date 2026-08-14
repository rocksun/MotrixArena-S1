"""Grid-search two Section01 reward scales and track every training run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

ENV_NAME = "vbot_navigation_section01"
TWO_PLACES = Decimal("0.01")
RESULT_FIELDS = (
    "index",
    "total",
    "per_leg_swing",
    "swing_foot_height",
    "status",
    "return_code",
    "started_at",
    "finished_at",
    "duration_seconds",
    "run_dir",
    "log_file",
    "command",
)


def decimal_arg(value: str) -> Decimal:
    """Parse a command-line decimal with at most two fractional digits."""
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"not a decimal number: {value}") from exc
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError(f"must be finite: {value}")
    if parsed != parsed.quantize(TWO_PLACES):
        raise argparse.ArgumentTypeError(f"must have at most two decimal places: {value}")
    return parsed.quantize(TWO_PLACES)


def inclusive_range(start: Decimal, stop: Decimal, step: Decimal) -> list[Decimal]:
    """Return an inclusive, precisely stepped decimal range."""
    if step <= 0:
        raise ValueError("step must be greater than 0")
    if start > stop:
        raise ValueError("start must be less than or equal to stop")
    if (stop - start) % step:
        raise ValueError("step must land exactly on stop")

    values = []
    value = start
    while value <= stop:
        values.append(value.quantize(TWO_PLACES))
        value += step
    return values


def format_command(command: list[str]) -> str:
    """Format a subprocess argument list for human-readable logs."""
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def append_result(path: Path, result: dict[str, object]) -> None:
    """Append and flush one result so progress survives interruption."""
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=RESULT_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(result)
        result_file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Traverse all per_leg_swing and swing_foot_height combinations, "
            "run Section01 training sequentially, and record each result."
        )
    )
    parser.add_argument(
        "--per-leg-swing-range",
        nargs=3,
        required=True,
        metavar=("START", "STOP", "STEP"),
        type=decimal_arg,
        help="Inclusive per_leg_swing range (values may have at most 2 decimal places)",
    )
    parser.add_argument(
        "--swing-foot-height-range",
        nargs=3,
        required=True,
        metavar=("START", "STOP", "STEP"),
        type=decimal_arg,
        help="Inclusive swing_foot_height range (values may have at most 2 decimal places)",
    )
    parser.add_argument(
        "--sweep-id",
        help="Output directory name; defaults to a timestamp",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failed training (default: continue)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all combinations and commands without starting training",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to train.py; place them after --",
    )
    args = parser.parse_args()
    if args.train_args[:1] == ["--"]:
        args.train_args = args.train_args[1:]
    return args


def main() -> int:
    args = parse_args()
    try:
        per_leg_values = inclusive_range(*args.per_leg_swing_range)
        foot_height_values = inclusive_range(*args.swing_foot_height_range)
    except ValueError as exc:
        print(f"argument error: {exc}", file=sys.stderr, flush=True)
        return 2

    project_root = Path(__file__).resolve().parent.parent
    sweep_id = args.sweep_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if Path(sweep_id).name != sweep_id or sweep_id in {".", ".."}:
        print("argument error: --sweep-id must be a single directory name", file=sys.stderr, flush=True)
        return 2

    sweep_root = project_root / "runs" / "sweeps" / ENV_NAME / sweep_id
    logs_dir = sweep_root / "logs"
    results_path = sweep_root / "results.csv"
    combinations = [(per_leg, foot_height) for per_leg in per_leg_values for foot_height in foot_height_values]
    total = len(combinations)

    print("=" * 88, flush=True)
    print(f"Section01 reward sweep: {sweep_id}", flush=True)
    print(
        f"per_leg_swing: {per_leg_values[0]:.2f} .. {per_leg_values[-1]:.2f} "
        f"({len(per_leg_values)} values)",
        flush=True,
    )
    print(
        f"swing_foot_height: {foot_height_values[0]:.2f} .. {foot_height_values[-1]:.2f} "
        f"({len(foot_height_values)} values)",
        flush=True,
    )
    print(f"Total training runs: {total}", flush=True)
    print(f"Sweep directory: {sweep_root}", flush=True)
    print("=" * 88, flush=True)

    if not args.dry_run:
        if sweep_root.exists():
            print(
                f"output error: sweep directory already exists: {sweep_root}",
                file=sys.stderr,
                flush=True,
            )
            return 2
        logs_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "sweep_id": sweep_id,
            "environment": ENV_NAME,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "per_leg_swing": [f"{value:.2f}" for value in per_leg_values],
            "swing_foot_height": [f"{value:.2f}" for value in foot_height_values],
            "total": total,
            "extra_train_args": args.train_args,
        }
        (sweep_root / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    successful = 0
    failed = 0
    sweep_started = time.monotonic()

    for index, (per_leg, foot_height) in enumerate(combinations, start=1):
        combination_name = f"per_leg_swing_{per_leg:.2f}__swing_foot_height_{foot_height:.2f}"
        relative_run_dir = Path(f"sweep_{sweep_id}__{combination_name}")
        run_dir = project_root / "runs" / ENV_NAME / relative_run_dir
        log_file = logs_dir / f"{index:04d}_{combination_name}.log"
        command = [
            "uv",
            "run",
            "scripts/train.py",
            "--env",
            ENV_NAME,
            "--per-leg-swing",
            f"{per_leg:.2f}",
            "--swing-foot-height",
            f"{foot_height:.2f}",
            "--experiment-name",
            relative_run_dir.as_posix(),
            *args.train_args,
        ]
        command_text = format_command(command)

        elapsed = time.monotonic() - sweep_started
        average = elapsed / (index - 1) if index > 1 else 0.0
        eta = average * (total - index + 1)
        print("\n" + "-" * 88, flush=True)
        print(
            f"[{index}/{total}] per_leg_swing={per_leg:.2f}, "
            f"swing_foot_height={foot_height:.2f}",
            flush=True,
        )
        print(f"Progress: {(index - 1) / total:.1%} complete | elapsed={elapsed:.0f}s | ETA={eta:.0f}s", flush=True)
        print(f"Run directory: {run_dir}", flush=True)
        print(f"Training log: {log_file}", flush=True)
        print(f"Command: {command_text}", flush=True)

        if args.dry_run:
            continue

        started_at = datetime.now().astimezone()
        started_monotonic = time.monotonic()
        return_code = -1
        status = "failed"
        process = None

        with log_file.open("w", encoding="utf-8", errors="replace") as output:
            output.write(f"Started: {started_at.isoformat(timespec='seconds')}\n")
            output.write(f"Command: {command_text}\n")
            output.flush()
            try:
                process = subprocess.Popen(
                    command,
                    cwd=project_root,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    output.write(line)
                    output.flush()
                return_code = process.wait()
                status = "success" if return_code == 0 else "failed"
            except KeyboardInterrupt:
                status = "interrupted"
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        return_code = process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        return_code = process.wait()
            except OSError as exc:
                output.write(f"Failed to start training: {exc}\n")
                print(f"Failed to start training: {exc}", file=sys.stderr, flush=True)

        finished_at = datetime.now().astimezone()
        duration = time.monotonic() - started_monotonic
        append_result(
            results_path,
            {
                "index": index,
                "total": total,
                "per_leg_swing": f"{per_leg:.2f}",
                "swing_foot_height": f"{foot_height:.2f}",
                "status": status,
                "return_code": return_code,
                "started_at": started_at.isoformat(timespec="seconds"),
                "finished_at": finished_at.isoformat(timespec="seconds"),
                "duration_seconds": f"{duration:.1f}",
                "run_dir": str(run_dir),
                "log_file": str(log_file),
                "command": command_text,
            },
        )
        print(
            f"[{index}/{total}] {status.upper()} (exit={return_code}, duration={duration:.1f}s); "
            f"result appended to {results_path}",
            flush=True,
        )

        if status == "success":
            successful += 1
        else:
            failed += 1
        if status == "interrupted":
            print("Sweep interrupted by user.", flush=True)
            return 130
        if status == "failed" and args.stop_on_error:
            print("Stopping after the first failure as requested.", flush=True)
            break

    if args.dry_run:
        print(f"\nDry run complete: {total} commands checked; no files were written.", flush=True)
        return 0

    duration = time.monotonic() - sweep_started
    print("\n" + "=" * 88, flush=True)
    print(
        f"Sweep finished: successful={successful}, failed={failed}, "
        f"elapsed={duration:.1f}s, results={results_path}",
        flush=True,
    )
    print("=" * 88, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
