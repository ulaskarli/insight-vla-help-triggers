"""Run an uncertainty-enabled OpenPI policy behind an INSIGHT Strong gate.

This script is intentionally kept outside the core ``insight`` package because
GELLO and OpenPI are optional robot-runtime dependencies.

Modes
-----
inspect:
    Query the VLA once, print/save the INSIGHT decision, and never move the robot.
autonomous:
    Query a fresh VLA chunk, score it once with INSIGHT, and execute the chunk
    only when INSIGHT does not request help.
intervention:
    Same as autonomous, but a halt can hand control to GELLO and record a human
    recovery demonstration beginning from the halted state.

The gate sits *before* action chunk buffering: no action from a rejected VLA
chunk is sent to the robot.
"""

from __future__ import annotations

import datetime as dt
import enum
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tyro

from insight.strong_trigger import StrongHelpTrigger

from gello.agents.gello_agent import GelloAgent
from gello.data_utils.format_obs import save_frame
from gello.env import RobotEnv
from gello.zmq_core.camera_node import ZMQClientCamera
from gello.zmq_core.robot_node import ZMQClientRobot
from openpi_client import websocket_client_policy


class RunMode(enum.Enum):
    INSPECT = "inspect"
    AUTONOMOUS = "autonomous"
    INTERVENTION = "intervention"


@dataclass
class Args:
    mode: RunMode = RunMode.INSPECT

    # Robot / camera ZMQ endpoints.
    robot_host: str = "127.0.0.1"
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hz: int = 30

    # OpenPI policy server. Use a connectable address, not 0.0.0.0.
    policy_host: str = "127.0.0.1"
    policy_port: int = 8000
    prompt: str = "lift the corn"
    action_horizon: int = 30

    # INSIGHT Strong. Point this at single_strong or single_strong_jumbo.
    checkpoint_dir: str = "experiments/results/single_strong_jumbo"
    checkpoint_pattern: str = "single_fold*.pt"
    threshold_logit: float = 0.5
    trim_head: int = 3
    trim_tail: int = 2
    device: Optional[str] = None

    # Command safety. Set <=0 to disable delta clipping.
    max_joint_delta: float = 0.05

    # Optional GELLO takeover after an INSIGHT halt.
    gello_port: Optional[str] = None
    intervention_dir: str = "~/insight_interventions"

    # Optional explicit start pose; omitted by default to avoid an automatic move.
    home_joints: Optional[Tuple[float, ...]] = None
    home_steps: int = 30


def make_policy_observation(obs: dict, prompt: str) -> dict:
    return {
        "observation/state": np.asarray(obs["joint_positions"]),
        "observation/image": np.asarray(obs["base_rgb"]),
        "observation/wrist_image": np.asarray(obs["wrist_rgb"]),
        "prompt": prompt,
    }


def clip_joint_target(target: np.ndarray, current: np.ndarray, max_delta: float) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if max_delta <= 0:
        return target
    delta = target - current
    peak = float(np.max(np.abs(delta)))
    if peak > max_delta:
        delta = delta / peak * max_delta
    return current + delta


def move_to_home(env: RobotEnv, home: Tuple[float, ...], steps: int) -> dict:
    obs = env.get_obs()
    current = np.asarray(obs["joint_positions"], dtype=np.float64)
    target = np.asarray(home, dtype=np.float64)
    if current.shape != target.shape:
        raise ValueError(f"home_joints shape {target.shape} != robot state {current.shape}")
    for q in np.linspace(current, target, max(1, steps)):
        obs = env.step(q)
    return obs


def save_halt_snapshot(
    root: Path,
    *,
    obs: dict,
    policy_output: dict,
    features: np.ndarray,
    decision,
    prompt: str,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = root / f"halt_{stamp}"
    path.mkdir(parents=True, exist_ok=False)

    np.savez_compressed(
        path / "halt_state.npz",
        joint_positions=np.asarray(obs["joint_positions"]),
        base_rgb=np.asarray(obs["base_rgb"]),
        wrist_rgb=np.asarray(obs["wrist_rgb"]),
        rejected_actions=np.asarray(policy_output["actions"]),
        insight_features=features,
        member_logits=np.asarray(decision.member_logits, dtype=np.float32),
    )
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "prompt": prompt,
                "should_help": bool(decision.should_help),
                "logit": float(decision.logit),
                "probability": float(decision.probability),
                "num_tokens": int(decision.num_tokens),
                "feature_order": ["au", "eu", "entropy", "chosen_token_log_probability"],
                "historical_source_field_for_logp": "perplexity",
            },
            indent=2,
        )
    )
    return path


def run_gello_recovery(env: RobotEnv, args: Args, halt_dir: Path) -> dict:
    """Teleoperate from the current halted state and save the recovery demo."""
    if not args.gello_port:
        raise ValueError("--gello-port is required for intervention mode.")

    from gello.data_utils.keyboard_interface import KBReset

    obs = env.get_obs()
    current = np.asarray(obs["joint_positions"], dtype=np.float64)

    print("\nINSIGHT halted the VLA before executing the rejected chunk.")
    print("Align the GELLO leader with the halted robot pose, then press Enter.")
    input()

    # Passing the current robot state as start_joints makes the takeover begin
    # from the halted configuration rather than an unrelated reset pose.
    agent = GelloAgent(port=args.gello_port, start_joints=current)
    keyboard = KBReset()

    demo_dir = halt_dir / "gello_demo"
    print("Press 's' to start the recovery demonstration; press 'q' to finish it.")
    recording = False

    while True:
        state = keyboard.update()
        if not recording:
            if state == "start":
                demo_dir.mkdir(parents=True, exist_ok=True)
                recording = True
                print(f"Recording recovery demo to {demo_dir}")
            else:
                time.sleep(0.02)
                continue
        elif state == "normal":
            print("Recovery demonstration complete.")
            break

        obs = env.get_obs()
        target = np.asarray(agent.act(obs), dtype=np.float64)
        current = np.asarray(obs["joint_positions"], dtype=np.float64)
        command = clip_joint_target(target, current, args.max_joint_delta)
        now = dt.datetime.now()
        save_frame(demo_dir, now, obs, command)
        obs = env.step(command)

    return env.get_obs()


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    trigger = StrongHelpTrigger.from_directory(
        args.checkpoint_dir,
        pattern=args.checkpoint_pattern,
        threshold_logit=args.threshold_logit,
        trim_head=args.trim_head,
        trim_tail=args.trim_tail,
        device=args.device,
    )
    logging.info(
        "Loaded %d INSIGHT Strong checkpoint(s) from %s on %s",
        len(trigger.models),
        args.checkpoint_dir,
        trigger.device,
    )

    cameras = {
        "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.robot_host),
        "base": ZMQClientCamera(port=args.base_camera_port, host=args.robot_host),
    }
    robot = ZMQClientRobot(port=args.robot_port, host=args.robot_host)
    env = RobotEnv(robot, control_rate_hz=args.hz, camera_dict=cameras)

    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.policy_host,
        port=args.policy_port,
    )
    logging.info("OpenPI server metadata: %s", policy.get_server_metadata())

    if args.home_joints is not None:
        obs = move_to_home(env, args.home_joints, args.home_steps)
    else:
        obs = env.get_obs()

    intervention_root = Path(args.intervention_dir).expanduser()

    while True:
        raw = policy.infer(make_policy_observation(obs, args.prompt))
        features = trigger.features_from_policy_output(raw)
        decision = trigger.predict_features(features)
        actions = np.asarray(raw["actions"])

        print(
            f"INSIGHT: logit={decision.logit:.4f} "
            f"prob={decision.probability:.4f} "
            f"tokens={decision.num_tokens} "
            f"decision={'HELP/HALT' if decision.should_help else 'EXECUTE'}"
        )
        print(
            "Returned shapes:",
            {k: np.asarray(v).shape for k, v in raw.items() if hasattr(v, "__array__")},
        )

        if args.mode == RunMode.INSPECT:
            halt_dir = save_halt_snapshot(
                intervention_root / "inspection",
                obs=obs,
                policy_output=raw,
                features=features,
                decision=decision,
                prompt=args.prompt,
            )
            print(f"Inspection snapshot saved to {halt_dir}. No robot action was executed.")
            return

        if decision.should_help:
            halt_dir = save_halt_snapshot(
                intervention_root,
                obs=obs,
                policy_output=raw,
                features=features,
                decision=decision,
                prompt=args.prompt,
            )
            print(f"Rejected VLA chunk saved to {halt_dir}")

            while True:
                options = "[r] re-infer, [q] quit"
                if args.mode == RunMode.INTERVENTION:
                    options = "[g] GELLO recovery, " + options
                choice = input(f"HALTED — {options}: ").strip().lower()
                if choice == "q":
                    return
                if choice == "r":
                    obs = env.get_obs()
                    break
                if choice == "g" and args.mode == RunMode.INTERVENTION:
                    obs = run_gello_recovery(env, args, halt_dir)
                    # Always discard the rejected VLA chunk and re-infer after
                    # the human changes the state.
                    break
            continue

        if actions.ndim != 2:
            raise ValueError(f"Expected actions [H,D], got {actions.shape}")

        for target in actions[: args.action_horizon]:
            current_obs = env.get_obs()
            current = np.asarray(current_obs["joint_positions"], dtype=np.float64)
            command = clip_joint_target(target, current, args.max_joint_delta)
            obs = env.step(command)


if __name__ == "__main__":
    main(tyro.cli(Args))
