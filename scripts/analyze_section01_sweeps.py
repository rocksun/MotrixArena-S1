"""Analyze Section01 ``sweep_*`` TensorBoard runs and rank their route progress."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ENV_NAME = "vbot_navigation_section01"
RUN_NAME_PATTERN = re.compile(
    r"^sweep_(?P<sweep_id>.+?)__per_leg_swing_(?P<per_leg_swing>-?\d+(?:\.\d+)?)"
    r"__swing_foot_height_(?P<swing_foot_height>-?\d+(?:\.\d+)?)$"
)

# Keep these synchronized with VBotSection01EnvCfg. They describe the route used
# by the existing sweep runs, not a new evaluation trajectory.
START_Y = -2.40
WAYPOINT_Y = (-0.60, 1.20, 2.25, 4.00, 6.00, 7.00, 7.80)
REACH_THRESHOLD_M = 0.45

GOAL_INDEX_TAG = "metrics / goal_idx (max)"
WAYPOINT_DISTANCE_TAG = "metrics / distance_to_waypoint (min)"
SUCCESS_RATE_TAG = "metrics / goal_success_rate (mean)"
WINDOW_SUCCESS_RATE_TAG = "metrics / goal_success_rate_window200 (mean)"
REACH_ALL_TAG = "Reward Instant / reach_all_goal (max)"
TOTAL_REWARD_TAG = "Reward / Total reward (mean)"

CSV_FIELDS = (
    "rank",
    "sweep_id",
    "per_leg_swing",
    "swing_foot_height",
    "achieved_distance_m",
    "confirmed_lower_bound_m",
    "completed_waypoints",
    "total_waypoints",
    "max_goal_idx_metric",
    "all_goals_reached",
    "max_goal_success_rate",
    "latest_goal_success_rate",
    "max_window200_success_rate",
    "latest_window200_success_rate",
    "min_logged_waypoint_distance_m",
    "best_mean_total_reward",
    "latest_mean_total_reward",
    "last_event_step",
    "training_status",
    "event_file_count",
    "best_checkpoint",
    "run_dir",
    "error",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    default_root = project_root / "runs" / ENV_NAME
    default_output = (
        project_root
        / "runs"
        / "sweeps"
        / ENV_NAME
        / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Read TensorBoard data from Section01 sweep_* runs, estimate the farthest "
            "confirmed route distance, rank the runs, and export a CSV report."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=default_root,
        help=f"Directory containing sweep_* runs (default: {default_root})",
    )
    parser.add_argument(
        "--pattern",
        default="sweep_*",
        help="Run directory glob, useful for selecting one sweep ID (default: sweep_*)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"CSV output path (default: {default_output})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of ranked rows printed to the console; 0 prints all (default: 20)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --output if it already exists",
    )
    args = parser.parse_args()
    if args.top < 0:
        parser.error("--top must be greater than or equal to 0")
    return args


def normalized_path(path: Path) -> str:
    """Create a case-insensitive path key suitable for Windows manifests."""
    return str(path.resolve()).replace("\\", "/").casefold()


def load_training_statuses(project_root: Path) -> dict[str, str]:
    """Load status values written by sweep_section01_rewards.py."""
    status_by_run: dict[str, str] = {}
    manifest_root = project_root / "runs" / "sweeps" / ENV_NAME
    if not manifest_root.exists():
        return status_by_run

    for results_file in manifest_root.glob("*/results.csv"):
        try:
            with results_file.open("r", newline="", encoding="utf-8-sig") as stream:
                for row in csv.DictReader(stream):
                    run_dir = row.get("run_dir", "").strip()
                    if run_dir:
                        status_by_run[normalized_path(Path(run_dir))] = row.get("status", "unknown")
        except (OSError, csv.Error):
            continue
    return status_by_run


def scalar_values(accumulator: EventAccumulator, tag: str) -> list[Any]:
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return accumulator.Scalars(tag)


def scalar_summary(accumulator: EventAccumulator, tag: str) -> tuple[float | None, float | None]:
    events = scalar_values(accumulator, tag)
    finite_events = [event for event in events if math.isfinite(event.value)]
    if not finite_events:
        return None, None
    return max(event.value for event in finite_events), finite_events[-1].value


def infer_route_progress(max_goal_idx: float | None, all_goals_reached: bool) -> tuple[int, float, float]:
    """Infer completed waypoints and distance from the aggregated goal-index metric."""
    if all_goals_reached:
        completed = len(WAYPOINT_Y)
    elif max_goal_idx is None:
        completed = 0
    else:
        # The logged scalar is the time-window mean of a per-step integer maximum.
        # A positive fractional part means the next integer index occurred in that window.
        highest_active_index = max(0, math.ceil(max_goal_idx - 1e-6))
        completed = min(highest_active_index, len(WAYPOINT_Y) - 1)

    if completed == 0:
        return 0, 0.0, 0.0

    reached_y = WAYPOINT_Y[completed - 1]
    nominal_distance = reached_y - START_Y
    conservative_distance = max(0.0, nominal_distance - REACH_THRESHOLD_M)
    return completed, nominal_distance, conservative_distance


def find_best_checkpoint(run_dir: Path) -> str:
    checkpoint_dir = run_dir / "checkpoints"
    best = sorted(checkpoint_dir.glob("best_agent.*")) if checkpoint_dir.exists() else []
    if best:
        return str(best[-1].resolve())

    numbered = sorted(
        checkpoint_dir.glob("agent_*.*"),
        key=lambda path: int(re.search(r"agent_(\d+)", path.stem).group(1))
        if re.search(r"agent_(\d+)", path.stem)
        else -1,
    )
    return str(numbered[-1].resolve()) if numbered else ""


def analyze_run(run_dir: Path, status_by_run: dict[str, str]) -> dict[str, Any]:
    match = RUN_NAME_PATTERN.fullmatch(run_dir.name)
    base_row: dict[str, Any] = {
        "rank": "",
        "sweep_id": match.group("sweep_id") if match else "",
        "per_leg_swing": match.group("per_leg_swing") if match else "",
        "swing_foot_height": match.group("swing_foot_height") if match else "",
        "training_status": status_by_run.get(normalized_path(run_dir), "unknown_or_running"),
        "event_file_count": 0,
        "best_checkpoint": find_best_checkpoint(run_dir),
        "run_dir": str(run_dir.resolve()),
        "error": "",
    }

    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    base_row["event_file_count"] = len(event_files)
    if not event_files:
        base_row["error"] = "no TensorBoard event file"
        return base_row

    try:
        accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        accumulator.Reload()

        max_goal_idx, _ = scalar_summary(accumulator, GOAL_INDEX_TAG)
        max_success, latest_success = scalar_summary(accumulator, SUCCESS_RATE_TAG)
        max_window_success, latest_window_success = scalar_summary(accumulator, WINDOW_SUCCESS_RATE_TAG)
        min_distance_events = scalar_values(accumulator, WAYPOINT_DISTANCE_TAG)
        finite_distances = [event.value for event in min_distance_events if math.isfinite(event.value)]
        min_logged_distance = min(finite_distances) if finite_distances else None
        reach_all_events = scalar_values(accumulator, REACH_ALL_TAG)
        all_goals_reached = any(event.value > 0 for event in reach_all_events)
        best_reward, latest_reward = scalar_summary(accumulator, TOTAL_REWARD_TAG)

        scalar_tags = accumulator.Tags().get("scalars", [])
        last_event_step = max(
            (events[-1].step for tag in scalar_tags if (events := accumulator.Scalars(tag))),
            default=0,
        )
        completed, achieved_distance, lower_bound = infer_route_progress(max_goal_idx, all_goals_reached)

        base_row.update(
            {
                "achieved_distance_m": achieved_distance,
                "confirmed_lower_bound_m": lower_bound,
                "completed_waypoints": completed,
                "total_waypoints": len(WAYPOINT_Y),
                "max_goal_idx_metric": max_goal_idx,
                "all_goals_reached": all_goals_reached,
                "max_goal_success_rate": max_success,
                "latest_goal_success_rate": latest_success,
                "max_window200_success_rate": max_window_success,
                "latest_window200_success_rate": latest_window_success,
                "min_logged_waypoint_distance_m": min_logged_distance,
                "best_mean_total_reward": best_reward,
                "latest_mean_total_reward": latest_reward,
                "last_event_step": last_event_step,
            }
        )
    except Exception as exc:  # TensorBoard raises several backend-specific data errors.
        base_row["error"] = f"{type(exc).__name__}: {exc}"

    return base_row


def numeric_sort_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else float("-inf")


def format_csv_value(field: str, value: Any) -> Any:
    if value is None:
        return ""
    if field in {
        "achieved_distance_m",
        "confirmed_lower_bound_m",
        "max_goal_idx_metric",
        "min_logged_waypoint_distance_m",
        "best_mean_total_reward",
        "latest_mean_total_reward",
    } and isinstance(value, (int, float)):
        return f"{value:.4f}"
    if "success_rate" in field and isinstance(value, (int, float)):
        return f"{value:.6f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return value


def write_report(output: Path, rows: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_csv_value(field, row.get(field)) for field in CSV_FIELDS})


def print_ranking(rows: list[dict[str, Any]], top: int) -> None:
    valid_rows = [row for row in rows if not row.get("error")]
    shown = valid_rows if top == 0 else valid_rows[:top]
    print("\nRank  per_leg  foot_h  distance  lower_bd  waypoints  success(max)  step      status")
    print("-" * 96)
    for row in shown:
        success = row.get("max_goal_success_rate")
        success_text = f"{success:.4f}" if isinstance(success, (int, float)) else "n/a"
        print(
            f"{row['rank']:>4}  {row['per_leg_swing']:>7}  {row['swing_foot_height']:>6}  "
            f"{row['achieved_distance_m']:>8.2f}  {row['confirmed_lower_bound_m']:>8.2f}  "
            f"{row['completed_waypoints']:>4}/{row['total_waypoints']:<4}  {success_text:>12}  "
            f"{row['last_event_step']:>8}  {row['training_status']}"
        )

    error_rows = [row for row in rows if row.get("error")]
    if error_rows:
        print(f"\nRuns with analysis errors: {len(error_rows)}")
        for row in error_rows:
            print(f"  {Path(row['run_dir']).name}: {row['error']}")


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    runs_dir = args.runs_dir.resolve()
    output = args.output if args.output.is_absolute() else (project_root / args.output)
    output = output.resolve()

    if not runs_dir.is_dir():
        print(f"error: runs directory does not exist: {runs_dir}", file=sys.stderr)
        return 2
    if output.exists() and not args.force:
        print(f"error: output already exists (use --force): {output}", file=sys.stderr)
        return 2

    run_dirs = sorted(path for path in runs_dir.glob(args.pattern) if path.is_dir())
    if not run_dirs:
        print(f"error: no run directories matched {args.pattern!r} below {runs_dir}", file=sys.stderr)
        return 2

    print(f"Found {len(run_dirs)} sweep run(s) below {runs_dir}")
    print("Distance definition: farthest confirmed waypoint relative to nominal start Y=-2.40 m")
    print(f"Reach threshold: {REACH_THRESHOLD_M:.2f} m; conservative lower bound is also reported")

    statuses = load_training_statuses(project_root)
    rows = []
    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[{index}/{len(run_dirs)}] Reading {run_dir.name}", flush=True)
        rows.append(analyze_run(run_dir, statuses))

    rows.sort(
        key=lambda row: (
            numeric_sort_value(row, "achieved_distance_m"),
            numeric_sort_value(row, "max_goal_success_rate"),
            numeric_sort_value(row, "best_mean_total_reward"),
        ),
        reverse=True,
    )
    rank = 0
    for row in rows:
        if not row.get("error"):
            rank += 1
            row["rank"] = rank

    write_report(output, rows)
    print_ranking(rows, args.top)
    print(f"\nCSV report: {output}")
    return 1 if all(row.get("error") for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
