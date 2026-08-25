"""Run one-at-a-time reward-scale sweeps for vbot_navigation_section01.

Examples:
    # For every configured reward, train with its original value and +10% value.
    uv run --no-sync scripts/sweep_reward_scales.py 0 0.1

    # Sweep only two rewards, forwarding an additional option to train.py.
    uv run --no-sync scripts/sweep_reward_scales.py 0 -0.1 0.1 \
        --parameters tracking_lin_vel,feet_air_time --train-arg=--seed=42

Each run changes exactly one reward scale.  The script completes every requested
change for one reward before continuing to the next reward, and restores cfg.py
when it exits (also after Ctrl+C or a failed run).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "motrix_envs" / "src" / "motrix_envs" / "navigation" / "cfg.py"
DEFAULT_ENV = "vbot_navigation_section01"

# Keep this order aligned with the configuration so sweep results are predictable.
DEFAULT_REWARDS = (
    "termination",
    "tracking_lin_vel",
    "tracking_ang_vel",
    "tracking_goal_vel",
    "tracking_yaw",
    "forward_progress",
    "target_progress",
    "reach_goal",
    "reach_all_goal",
    "lin_vel_z",
    "ang_vel_xy",
    "orientation",
    "torques",
    "dof_vel",
    "dof_acc",
    "action_rate",
    "feet_air_time",
    "anti_stall",
    "dof_pos_limits",
    "undesired_contacts",
    "per_leg_swing",
    "gait_symmetry",
    "energy",
    "swing_foot_height",
    "drop_leg_catchup",
    "drop_pitch",
)

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
SCALES_BLOCK = re.compile(
    r"(?P<prefix>class RewardConfig:.*?default_factory=lambda:\s*\{\n)"
    r"(?P<body>.*?)"
    r"(?P<suffix>^\s{12}\}\s*\n\s*\)\s*\n)",
    re.DOTALL | re.MULTILINE,
)


@dataclass(frozen=True)
class SweepRun:
    reward: str
    change: float
    original_value: float

    @property
    def new_value(self) -> float:
        return self.original_value * (1.0 + self.change)

    @property
    def experiment_name(self) -> str:
        timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        return f"{timestamp}__{self.reward}={format_number(self.new_value)}__PPO"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one reward scale at a time with relative changes."
    )
    parser.add_argument(
        "changes",
        nargs="+",
        type=float,
        help="Relative changes, e.g. 0 0.1 means unchanged and +10%%.",
    )
    parser.add_argument(
        "--parameters",
        default=",".join(DEFAULT_REWARDS),
        help="Comma-separated reward names to sweep (default: all listed rewards).",
    )
    parser.add_argument("--env", default=DEFAULT_ENV, help="Environment passed to train.py.")
    parser.add_argument(
        "--train-arg",
        action="append",
        default=[],
        help="Extra argument to pass to train.py; repeat this option as needed.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later runs if a training process exits unsuccessfully.",
    )
    args = parser.parse_args()
    if any(not math.isfinite(change) or change <= -1.0 for change in args.changes):
        parser.error("every change must be finite and greater than -1")
    return args


def selected_rewards(value: str) -> tuple[str, ...]:
    requested = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = tuple(name for name in requested if name not in DEFAULT_REWARDS)
    if unknown:
        raise ValueError(f"unsupported reward name(s): {', '.join(unknown)}")
    if not requested:
        raise ValueError("--parameters must name at least one reward")
    if len(set(requested)) != len(requested):
        raise ValueError("--parameters cannot contain duplicate reward names")
    # Preserve the config's stable order even if the command-line list is reordered.
    return tuple(name for name in DEFAULT_REWARDS if name in requested)


def get_scales_body(source: str) -> tuple[re.Match[str], str]:
    # cfg.py also has a module-level RewardConfig for another navigation setup.
    # Identify the section01 nested table by a key unique to this sweep instead
    # of replacing the first class named RewardConfig.
    for match in SCALES_BLOCK.finditer(source):
        body = match.group("body")
        if '"tracking_lin_vel"' in body and '"reach_all_goal"' in body:
            return match, body
    raise RuntimeError(f"could not find section01 RewardConfig.scales in {CONFIG_PATH}")


def reward_value(source: str, reward: str) -> float:
    _, body = get_scales_body(source)
    match = re.search(
        rf'^[ \t]*"{re.escape(reward)}"[ \t]*:[ \t]*({NUMBER})[ \t]*,',
        body,
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"could not find reward scale {reward!r} in RewardConfig.scales")
    return float(match.group(1))


def format_number(value: float) -> str:
    """Use a compact Python float literal while retaining enough sweep precision."""
    text = format(value, ".15g")
    return text if any(char in text for char in ".eE") else f"{text}.0"


def with_reward_value(source: str, reward: str, value: float) -> str:
    block, body = get_scales_body(source)
    pattern = re.compile(
        rf'(?P<prefix>^[ \t]*"{re.escape(reward)}"[ \t]*:[ \t]*){NUMBER}'
        r'(?P<suffix>[ \t]*,)',
        re.MULTILINE,
    )
    new_body, replacements = pattern.subn(
        lambda match: f"{match.group('prefix')}{format_number(value)}{match.group('suffix')}", body, count=1
    )
    if replacements != 1:
        raise RuntimeError(f"could not update reward scale {reward!r} in RewardConfig.scales")
    return source[: block.start("body")] + new_body + source[block.end("body") :]


def write_config(source: str) -> None:
    CONFIG_PATH.write_text(source, encoding="utf-8")


def run_training(args: argparse.Namespace, run: SweepRun) -> int:
    uv = shutil.which("uv") or "uv"
    command = [
        uv,
        "run",
        "--no-sync",
        "scripts/train.py",
        "--env",
        args.env,
        "--experiment-name",
        run.experiment_name,
        *args.train_arg,
    ]
    print(
        f"\n[{run.reward}] change={run.change:+.2%}: "
        f"{run.original_value:g} -> {run.new_value:g}",
        flush=True,
    )
    print("$ " + subprocess.list2cmdline(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def main() -> int:
    args = parse_args()
    rewards = selected_rewards(args.parameters)
    original_source = CONFIG_PATH.read_text(encoding="utf-8")
    expected_source: str | None = None
    failures: list[SweepRun] = []

    try:
        for reward in rewards:
            original_value = reward_value(original_source, reward)
            if original_value == 0.0 and any(change != 0.0 for change in args.changes):
                print(
                    f"Warning: {reward}=0.0; relative changes leave it at 0.0. "
                    "Use a non-zero base value before sweeping it.",
                    file=sys.stderr,
                )
            for change in args.changes:
                if expected_source is not None and CONFIG_PATH.read_text(encoding="utf-8") != expected_source:
                    raise RuntimeError(
                        f"{CONFIG_PATH} changed outside this script; refusing to overwrite it"
                    )
                run = SweepRun(reward=reward, change=change, original_value=original_value)
                # A zero change is the baseline run: leave the config text exactly as-is.
                expected_source = (
                    original_source
                    if change == 0
                    else with_reward_value(original_source, reward, run.new_value)
                )
                write_config(expected_source)
                returncode = run_training(args, run)
                if returncode:
                    print(f"Training failed with exit code {returncode}.", file=sys.stderr)
                    failures.append(run)
                    if not args.continue_on_error:
                        return returncode
    finally:
        if expected_source is not None and CONFIG_PATH.read_text(encoding="utf-8") == expected_source:
            write_config(original_source)
            print(f"Restored {CONFIG_PATH.relative_to(REPO_ROOT)}.", flush=True)
        elif expected_source is not None:
            print(
                f"Did not restore {CONFIG_PATH}: it was modified outside this script.",
                file=sys.stderr,
            )

    if failures:
        print(f"Completed with {len(failures)} failed run(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
