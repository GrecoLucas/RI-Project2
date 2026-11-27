"""Webots controller entry point for the Dynamic Sentinel Experiment.

Webots starts this file as the controller of the e-puck ``AGENT`` robot (which
has ``supervisor TRUE``). Arguments come from the robot's ``controllerArgs``
field in the world file, or from the command line when the controller is
started as an extern controller (``webots-controller``).

Modes:
    check  - random actions for a few episodes + Gymnasium API check (no training)
    train  - train PPO (optionally continue from --model)
    eval   - run a trained model deterministically and print statistics

Examples (controllerArgs):
    "--mode" "train" "--timesteps" "300000" "--sentinels" "0"
    "--mode" "eval" "--model" "models/ppo_sentinel_final.zip" "--episodes" "20"
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from controller import Supervisor  # provided by Webots

from config import EnvConfig, PPOConfig
from env import SentinelEscapeEnv, SimulationEnded

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
LOGS_DIR = REPO_ROOT / "logs"


def parse_args(argv):
    p = argparse.ArgumentParser(description="Dynamic Sentinel Experiment controller")
    p.add_argument("--mode", choices=["check", "train", "eval"], default="check")
    p.add_argument("--sentinels", type=int, default=None, help="override number of active sentinels")
    p.add_argument("--timesteps", type=int, default=None, help="override PPO total timesteps")
    p.add_argument("--model", type=str, default=None, help="model .zip to load (eval, or continue training)")
    p.add_argument("--episodes", type=int, default=10, help="episodes for check/eval")
    p.add_argument("--run-name", type=str, default="ppo_sentinel", help="prefix for saved models/logs")
    p.add_argument("--realtime", action="store_true", help="run in real-time instead of fast mode")
    p.add_argument("--quit", action="store_true", help="quit Webots when done (useful for headless runs)")
    p.add_argument("--log-file", type=str, default=None,
                   help="also write all output (incl. errors) to this file - useful for headless runs")
    return p.parse_args(argv)


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def run_check(env: SentinelEscapeEnv, episodes: int) -> None:
    from stable_baselines3.common.env_checker import check_env

    check_env(env, warn=True)
    print("[check] Gymnasium API check passed")
    for ep in range(episodes):
        obs, _ = env.reset()
        total, done, info = 0.0, False, {}
        while not done:
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            total += r
            done = term or trunc
        print(f"[check] episode {ep}: return={total:.2f} steps={info['steps']} "
              f"checkpoint={info['checkpoint_reached']} detected={info['detected']} success={info['is_success']}")


def run_train(env: SentinelEscapeEnv, args, ppo_cfg: PPOConfig) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    menv = Monitor(env, info_keywords=("is_success", "detected", "checkpoint_reached"))

    # TensorBoard is optional: without it SB3 would crash at learn(), so fall back to console logs.
    tb_log = str(LOGS_DIR)
    if importlib.util.find_spec("tensorboard") is None:
        print("[train] WARNING: tensorboard not installed -> no TensorBoard logs "
              "(pip install -r requirements.txt to enable them)")
        tb_log = None

    if args.model:
        print(f"[train] continuing from {args.model}")
        model = PPO.load(resolve(args.model), env=menv, device=ppo_cfg.device, tensorboard_log=tb_log)
    else:
        model = PPO(
            "MlpPolicy",
            menv,
            learning_rate=ppo_cfg.learning_rate,
            n_steps=ppo_cfg.n_steps,
            batch_size=ppo_cfg.batch_size,
            n_epochs=ppo_cfg.n_epochs,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            ent_coef=ppo_cfg.ent_coef,
            seed=ppo_cfg.seed,
            device=ppo_cfg.device,
            tensorboard_log=tb_log,
            verbose=1,
        )

    callback = CheckpointCallback(
        save_freq=ppo_cfg.checkpoint_every, save_path=str(MODELS_DIR), name_prefix=args.run_name
    )
    model.learn(
        total_timesteps=ppo_cfg.total_timesteps,
        callback=callback,
        tb_log_name=args.run_name,
        reset_num_timesteps=not bool(args.model),
    )
    out = MODELS_DIR / f"{args.run_name}_final.zip"
    model.save(out)
    print(f"[train] saved final model to {out}")


def run_eval(env: SentinelEscapeEnv, args) -> None:
    from stable_baselines3 import PPO

    if not args.model:
        raise SystemExit("[eval] --model is required")
    model = PPO.load(resolve(args.model), device="cpu")
    stats = {"success": 0, "detected": 0, "timeout": 0}
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done, total, info, trunc = False, 0.0, {}, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total += r
            done = term or trunc
        outcome = "success" if info["is_success"] else "detected" if info["detected"] else "timeout"
        stats[outcome] += 1
        print(f"[eval] episode {ep}: {outcome:8s} return={total:.2f} steps={info['steps']}")
    n = max(1, args.episodes)
    print("[eval] " + "  ".join(f"{k}={v}/{n} ({100 * v / n:.0f}%)" for k, v in stats.items()))


class _Tee:
    """Duplicate a stream into a log file (line-buffered)."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, data):
        self.stream.write(data)
        self.log.write(data)
        self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()


def main() -> None:
    args = parse_args(sys.argv[1:])
    if args.log_file:
        log_path = resolve(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "a", encoding="utf-8")
        sys.stdout = _Tee(sys.stdout, log)
        sys.stderr = _Tee(sys.stderr, log)
    robot = Supervisor()

    env_cfg, ppo_cfg = EnvConfig(), PPOConfig()
    if args.sentinels is not None:
        env_cfg.sentinels.num_sentinels = args.sentinels
    if args.timesteps is not None:
        ppo_cfg.total_timesteps = args.timesteps

    fast = args.mode == "train" and not args.realtime
    robot.simulationSetMode(
        Supervisor.SIMULATION_MODE_FAST if fast else Supervisor.SIMULATION_MODE_REAL_TIME
    )

    env = SentinelEscapeEnv(robot, env_cfg)
    print(f"[main] mode={args.mode} sentinels={env_cfg.sentinels.num_sentinels} "
          f"obs_dim={env.observation_space.shape[0]}")
    try:
        if args.mode == "check":
            run_check(env, args.episodes)
        elif args.mode == "train":
            run_train(env, args, ppo_cfg)
        else:
            run_eval(env, args)
    except SimulationEnded:
        print("[main] simulation ended by Webots")
        return
    finally:
        env.close()

    if args.quit:
        robot.simulationQuit(0)
    else:
        robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)


if __name__ == "__main__":
    main()
