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
    "best_goal_step",
    "steps_since_best_goal",
    "latest_goal_idx_metric",
    "recent_goal_slope_per_1k_steps",
    "recent_waypoint_distance_slope_per_1k_steps",
    "recent_success_rate_slope_per_1k_steps",
    "recent_reward_slope_per_1k_steps",
    "training_outlook",
    "continue_training_signal",
    "outlook_reason",
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
    parser.add_argument(
        "--trend-points",
        type=int,
        default=4,
        help="Number of latest TensorBoard points used for trend slopes (default: 4)",
    )
    parser.add_argument(
        "--stall-steps",
        type=int,
        default=5000,
        help="Steps without a new goal index before route progress is considered stalled (default: 5000)",
    )
    parser.add_argument(
        "--reward-flat-slope",
        type=float,
        default=25.0,
        help="Absolute reward slope per 1000 steps considered flat (default: 25)",
    )
    args = parser.parse_args()
    if args.top < 0:
        parser.error("--top must be greater than or equal to 0")
    if args.trend_points < 2:
        parser.error("--trend-points must be at least 2")
    if args.stall_steps <= 0:
        parser.error("--stall-steps must be greater than 0")
    if args.reward_flat_slope < 0:
        parser.error("--reward-flat-slope must be greater than or equal to 0")
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


def recent_slope(events: list[Any], point_count: int) -> float | None:
    """Return least-squares scalar change per 1000 training steps."""
    points = [(event.step / 1000.0, event.value) for event in events if math.isfinite(event.value)][-point_count:]
    if len(points) < 2:
        return None
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def best_goal_step(events: list[Any]) -> int | None:
    """Return the first step where the final maximum goal-index metric appeared."""
    finite_events = [event for event in events if math.isfinite(event.value)]
    if not finite_events:
        return None
    maximum = max(event.value for event in finite_events)
    return next(event.step for event in finite_events if event.value >= maximum - 1e-6)


def classify_training_outlook(
    *,
    all_goals_reached: bool,
    goal_point_count: int,
    steps_since_goal: int | None,
    goal_slope: float | None,
    distance_slope: float | None,
    success_slope: float | None,
    reward_slope: float | None,
    stall_steps: int,
    trend_points: int,
    reward_flat_slope: float,
) -> tuple[str, str, str]:
    """Classify whether more training is producing task progress."""
    if all_goals_reached:
        return "route_completed", "achieved", "At least one environment completed all route waypoints."
    if goal_point_count < trend_points or steps_since_goal is None:
        return (
            "insufficient_history",
            "insufficient",
            f"Only {goal_point_count} goal-progress points are available; need at least {trend_points}.",
        )

    goal_slope_value = goal_slope or 0.0
    distance_slope_value = distance_slope or 0.0
    success_slope_value = success_slope or 0.0
    reward_slope_value = reward_slope or 0.0
    details = (
        f"no new best goal for {steps_since_goal} steps; recent slopes per 1k steps: "
        f"goal={goal_slope_value:.4f}, waypoint_distance={distance_slope_value:.4f} m, "
        f"success={success_slope_value:.6f}, reward={reward_slope_value:.2f}"
    )

    recent_goal_steps = max(1000, stall_steps // 2)
    if goal_slope_value > 0.05 or steps_since_goal <= recent_goal_steps:
        return "route_progress_recent", "yes", details
    if steps_since_goal < stall_steps:
        return "monitor_more", "cautious", details
    if success_slope_value > 1e-4:
        return "success_rate_improving", "yes", details
    if distance_slope_value < -0.01:
        return "weak_distance_improvement", "cautious", details
    if reward_slope_value > reward_flat_slope:
        return (
            "route_stalled_reward_rising",
            "cautious",
            details + "; reward is rising without a new waypoint, so watch for reward-only optimization.",
        )
    if steps_since_goal >= stall_steps and abs(reward_slope_value) <= reward_flat_slope:
        return "likely_plateau", "no_or_adjust", details
    if steps_since_goal >= stall_steps and reward_slope_value < -reward_flat_slope:
        return "likely_regressing", "no_or_adjust", details
    return "uncertain", "cautious", details


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


def analyze_run(
    run_dir: Path,
    status_by_run: dict[str, str],
    *,
    trend_points: int,
    stall_steps: int,
    reward_flat_slope: float,
) -> dict[str, Any]:
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
        goal_events = scalar_values(accumulator, GOAL_INDEX_TAG)
        latest_goal_idx = goal_events[-1].value if goal_events else None
        max_success, latest_success = scalar_summary(accumulator, SUCCESS_RATE_TAG)
        max_window_success, latest_window_success = scalar_summary(accumulator, WINDOW_SUCCESS_RATE_TAG)
        min_distance_events = scalar_values(accumulator, WAYPOINT_DISTANCE_TAG)
        finite_distances = [event.value for event in min_distance_events if math.isfinite(event.value)]
        min_logged_distance = min(finite_distances) if finite_distances else None
        reach_all_events = scalar_values(accumulator, REACH_ALL_TAG)
        all_goals_reached = any(event.value > 0 for event in reach_all_events)
        best_reward, latest_reward = scalar_summary(accumulator, TOTAL_REWARD_TAG)
        reward_events = scalar_values(accumulator, TOTAL_REWARD_TAG)
        success_events = scalar_values(accumulator, WINDOW_SUCCESS_RATE_TAG)

        scalar_tags = accumulator.Tags().get("scalars", [])
        last_event_step = max(
            (events[-1].step for tag in scalar_tags if (events := accumulator.Scalars(tag))),
            default=0,
        )
        completed, achieved_distance, lower_bound = infer_route_progress(max_goal_idx, all_goals_reached)
        goal_best_step = best_goal_step(goal_events)
        steps_since_goal = last_event_step - goal_best_step if goal_best_step is not None else None
        goal_slope = recent_slope(goal_events, trend_points)
        distance_slope = recent_slope(min_distance_events, trend_points)
        success_slope = recent_slope(success_events, trend_points)
        reward_slope = recent_slope(reward_events, trend_points)
        outlook, continue_signal, outlook_reason = classify_training_outlook(
            all_goals_reached=all_goals_reached,
            goal_point_count=len(goal_events),
            steps_since_goal=steps_since_goal,
            goal_slope=goal_slope,
            distance_slope=distance_slope,
            success_slope=success_slope,
            reward_slope=reward_slope,
            stall_steps=stall_steps,
            trend_points=trend_points,
            reward_flat_slope=reward_flat_slope,
        )

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
                "best_goal_step": goal_best_step,
                "steps_since_best_goal": steps_since_goal,
                "latest_goal_idx_metric": latest_goal_idx,
                "recent_goal_slope_per_1k_steps": goal_slope,
                "recent_waypoint_distance_slope_per_1k_steps": distance_slope,
                "recent_success_rate_slope_per_1k_steps": success_slope,
                "recent_reward_slope_per_1k_steps": reward_slope,
                "training_outlook": outlook,
                "continue_training_signal": continue_signal,
                "outlook_reason": outlook_reason,
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
        "latest_goal_idx_metric",
        "recent_goal_slope_per_1k_steps",
        "recent_waypoint_distance_slope_per_1k_steps",
        "recent_reward_slope_per_1k_steps",
    } and isinstance(value, (int, float)):
        return f"{value:.4f}"
    if ("success_rate" in field or field == "recent_success_rate_slope_per_1k_steps") and isinstance(
        value, (int, float)
    ):
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
    print("\nRank  per_leg  foot_h  distance  waypoints  stalled  reward/1k  continue   outlook / status")
    print("-" * 124)
    for row in shown:
        stalled = row.get("steps_since_best_goal")
        stalled_text = str(stalled) if isinstance(stalled, int) else "n/a"
        reward_slope = row.get("recent_reward_slope_per_1k_steps")
        reward_slope_text = f"{reward_slope:.1f}" if isinstance(reward_slope, (int, float)) else "n/a"
        print(
            f"{row['rank']:>4}  {row['per_leg_swing']:>7}  {row['swing_foot_height']:>6}  "
            f"{row['achieved_distance_m']:>8.2f}  {row['completed_waypoints']:>4}/{row['total_waypoints']:<4}  "
            f"{stalled_text:>7}  {reward_slope_text:>9}  {row['continue_training_signal']:<9}  "
            f"{row['training_outlook']} / {row['training_status']}"
        )
        print(f"      {row['outlook_reason']}")

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
        rows.append(
            analyze_run(
                run_dir,
                statuses,
                trend_points=args.trend_points,
                stall_steps=args.stall_steps,
                reward_flat_slope=args.reward_flat_slope,
            )
        )

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
    outlook_counts: dict[str, int] = {}
    for row in rows:
        outlook = row.get("training_outlook")
        if outlook:
            outlook_counts[outlook] = outlook_counts.get(outlook, 0) + 1
    if outlook_counts:
        print("\nOutlook summary: " + ", ".join(f"{key}={value}" for key, value in sorted(outlook_counts.items())))
    print(f"\nCSV report: {output}")
    return 1 if all(row.get("error") for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
