"""Central configuration for the Dynamic Sentinel Experiment.

Every tunable number of the experiment lives here so that the world file,
the environment and the training script never disagree. Coordinates are in
metres, in the Webots world frame (ENU: x right, y forward, z up), with the
origin at the centre of the arena.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

Vec2 = Tuple[float, float]


@dataclass
class ArenaConfig:
    half_size: float = 1.0  # RectangleArena floorSize 2 x 2 -> walls at +-1.0 m
    spawn: Vec2 = (-0.8, -0.8)
    spawn_heading: float = 0.0  # radians, 0 = facing +x
    randomize_spawn: bool = False  # TODO (final goal): random spawn location
    checkpoint: Vec2 = (0.6, -0.6)
    checkpoint_radius: float = 0.10
    exit: Vec2 = (0.75, 0.75)
    exit_radius: float = 0.12
    randomize_exit: bool = False  # TODO (final goal): random exit location


@dataclass
class SentinelConfig:
    # How many sentinels are active. 0 -> intermediate goal 1, 1 -> intermediate goal 2.
    num_sentinels: int = 1
    # Size of the observation slots reserved for sentinels. Keep it fixed so a
    # policy trained with 0 sentinels can keep training with 1, 2, ... sentinels.
    max_sentinels: int = 3
    speed: float = 0.08  # m/s along the patrol route
    detection_range: float = 0.40  # m
    detection_half_angle: float = math.radians(30.0)  # half of the field of view
    # One looping patrol route per sentinel (sentinel i uses routes[i % len(routes)]).
    routes: List[List[Vec2]] = field(
        default_factory=lambda: [
            [(-0.6, 0.15), (0.6, 0.15), (0.6, 0.45), (-0.6, 0.45)],
            [(0.2, -0.3), (0.2, 0.7)],
            [(-0.5, -0.4), (-0.5, 0.6)],
        ]
    )


@dataclass
class RewardConfig:
    progress_scale: float = 10.0  # reward per metre of progress toward the current target
    checkpoint_bonus: float = 10.0
    exit_bonus: float = 25.0
    detected_penalty: float = -25.0
    wall_penalty: float = -0.5  # applied every step the robot is (almost) touching a wall
    step_penalty: float = -0.005  # small time pressure


@dataclass
class EnvConfig:
    control_period_ms: int = 64  # one RL step = this many ms of simulated time
    max_episode_steps: int = 1500  # ~96 s of simulated time
    max_wheel_speed: float = 6.28  # rad/s (e-puck motor limit)
    wall_ir_threshold: float = 0.20  # normalized IR value above which we count a wall contact
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    sentinels: SentinelConfig = field(default_factory=SentinelConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    total_timesteps: int = 300_000
    checkpoint_every: int = 25_000
    seed: int = 0
    device: str = "cpu"  # MLP policies train faster on CPU than on GPU
