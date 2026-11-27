"""Pure (Webots-free) logic: geometry, detection, observation and reward.

Nothing in this module imports Webots, so it can be unit-tested with plain
pytest (see ``tests/test_logic.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from config import RewardConfig

Vec2 = Tuple[float, float]

NUM_IR_SENSORS = 8
IR_MAX_VALUE = 4096.0  # e-puck proximity sensors return roughly 0..4095
SENTINEL_SLOT_SIZE = 5  # present flag, dx, dy, sin(rel heading), cos(rel heading)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def distance(a: Vec2, b: Vec2) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def to_robot_frame(robot_xy: Vec2, robot_heading: float, point_xy: Vec2) -> Vec2:
    """Express a world point in the robot frame (x forward, y left)."""
    dx = point_xy[0] - robot_xy[0]
    dy = point_xy[1] - robot_xy[1]
    c, s = math.cos(robot_heading), math.sin(robot_heading)
    return (c * dx + s * dy, -s * dx + c * dy)


def in_fov_cone(
    observer_xy: Vec2,
    observer_heading: float,
    target_xy: Vec2,
    max_range: float,
    half_angle: float,
) -> bool:
    """True if ``target_xy`` lies inside the observer's vision cone.

    Line-of-sight occlusion by obstacles is intentionally ignored for now.
    """
    d = distance(observer_xy, target_xy)
    if d > max_range:
        return False
    if d < 1e-9:
        return True
    bearing = math.atan2(target_xy[1] - observer_xy[1], target_xy[0] - observer_xy[0])
    return abs(wrap_angle(bearing - observer_heading)) <= half_angle


# --------------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------------- #
def observation_size(max_sentinels: int) -> int:
    # IR + (x, y, sin, cos) + (dist, sin, cos to target) + checkpoint flag + sentinel slots
    return NUM_IR_SENSORS + 4 + 3 + 1 + SENTINEL_SLOT_SIZE * max_sentinels


def build_observation(
    ir_values: Sequence[float],
    robot_xy: Vec2,
    robot_heading: float,
    target_xy: Vec2,
    checkpoint_reached: bool,
    sentinels: Sequence[Tuple[Vec2, float]],
    max_sentinels: int,
    arena_half_size: float,
) -> np.ndarray:
    """Build the continuous, roughly [-1, 1]-normalized observation vector."""
    obs = []

    # 1) proximity sensors
    ir = np.clip(np.asarray(ir_values, dtype=np.float64) / IR_MAX_VALUE, 0.0, 1.0)
    obs.extend(ir.tolist())

    # 2) own pose
    obs.extend(
        [
            robot_xy[0] / arena_half_size,
            robot_xy[1] / arena_half_size,
            math.sin(robot_heading),
            math.cos(robot_heading),
        ]
    )

    # 3) current target (checkpoint, then exit) in the robot frame
    diag = 2.0 * math.sqrt(2.0) * arena_half_size
    lx, ly = to_robot_frame(robot_xy, robot_heading, target_xy)
    bearing = math.atan2(ly, lx)
    obs.extend([math.hypot(lx, ly) / diag, math.sin(bearing), math.cos(bearing)])

    # 4) progress flag
    obs.append(1.0 if checkpoint_reached else 0.0)

    # 5) sentinels (fixed number of slots, zero-padded)
    for i in range(max_sentinels):
        if i < len(sentinels):
            s_xy, s_heading = sentinels[i]
            sx, sy = to_robot_frame(robot_xy, robot_heading, s_xy)
            rel = wrap_angle(s_heading - robot_heading)
            obs.extend(
                [1.0, sx / (2 * arena_half_size), sy / (2 * arena_half_size), math.sin(rel), math.cos(rel)]
            )
        else:
            obs.extend([0.0] * SENTINEL_SLOT_SIZE)

    return np.asarray(obs, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Reward
# --------------------------------------------------------------------------- #
@dataclass
class StepEvents:
    prev_target_dist: float
    target_dist: float
    reached_checkpoint: bool = False  # became true on this step
    reached_exit: bool = False
    detected: bool = False
    near_wall: bool = False


def compute_reward(ev: StepEvents, cfg: RewardConfig) -> Tuple[float, bool]:
    """Return ``(reward, terminated)`` for one step."""
    reward = cfg.step_penalty
    reward += cfg.progress_scale * (ev.prev_target_dist - ev.target_dist)
    if ev.near_wall:
        reward += cfg.wall_penalty
    if ev.reached_checkpoint:
        reward += cfg.checkpoint_bonus

    terminated = False
    if ev.detected:
        reward += cfg.detected_penalty
        terminated = True
    elif ev.reached_exit:
        reward += cfg.exit_bonus
        terminated = True
    return reward, terminated
