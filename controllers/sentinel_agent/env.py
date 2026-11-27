"""Gymnasium environment wrapping the Webots world.

The env does NOT create the Webots ``Supervisor`` itself; it receives one.
This keeps the class usable with a fake supervisor in unit tests and makes it
explicit that only one ``Supervisor`` instance may exist per controller
process.

Webots API calls used here (so a fake only needs these):
    robot.getBasicTimeStep(), robot.step(ms), robot.getSelf(), robot.getFromDef(name),
    robot.getRoot().getField("children").importMFNodeFromString(-1, vrml),
    robot.getDevice(name) -> motor (setPosition, setVelocity) / sensor (enable, getValue),
    node.getField(name).setSFVec3f / getSFVec3f / setSFRotation,
    node.getPosition(), node.getOrientation(), node.resetPhysics()
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import EnvConfig
from logic import (
    IR_MAX_VALUE,
    NUM_IR_SENSORS,
    StepEvents,
    build_observation,
    compute_reward,
    distance,
    in_fov_cone,
    observation_size,
)
from sentinels import WaypointSentinel

Vec2 = Tuple[float, float]


class SimulationEnded(RuntimeError):
    """Raised when Webots stops the simulation (robot.step returned -1)."""


def sentinel_vrml(def_name: str, range_m: float, half_angle: float) -> str:
    """VRML for a sentinel: red body + translucent field-of-view fan on the floor."""
    n = 8
    pts = ["0 0 0.003"]
    for k in range(n + 1):
        a = -half_angle + 2 * half_angle * k / n
        pts.append(f"{range_m * math.cos(a):.4f} {range_m * math.sin(a):.4f} 0.003")
    idx = " ".join(f"0 {k} {k + 1} -1" for k in range(1, n + 1))
    return f"""
DEF {def_name} Pose {{
  translation 0 0 0
  children [
    Pose {{
      translation 0 0 0.04
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.85 0.1 0.1 roughness 0.5 metalness 0 }}
          geometry Cylinder {{ height 0.08 radius 0.035 }}
        }}
      ]
    }}
    Pose {{
      translation 0.03 0 0.07
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 1 1 0 emissiveColor 1 1 0 metalness 0 }}
          geometry Box {{ size 0.02 0.03 0.015 }}
        }}
      ]
    }}
    Shape {{
      appearance PBRAppearance {{ baseColor 1 0.3 0.3 transparency 0.6 roughness 1 metalness 0 }}
      geometry IndexedFaceSet {{
        coord Coordinate {{ point [ {", ".join(pts)} ] }}
        coordIndex [ {idx} ]
        solid FALSE
      }}
    }}
  ]
}}
"""


def pad_vrml(def_name: str, radius: float, rgb: Tuple[float, float, float]) -> str:
    r, g, b = rgb
    return f"""
DEF {def_name} Pose {{
  translation 0 0 0.0005
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor {r} {g} {b} transparency 0.3 roughness 1 metalness 0 }}
      geometry Cylinder {{ height 0.001 radius {radius} }}
    }}
  ]
}}
"""


class SentinelEscapeEnv(gym.Env):
    """Robot must touch the checkpoint, then reach the exit, without being seen."""

    metadata = {"render_modes": []}

    def __init__(self, robot: Any, cfg: Optional[EnvConfig] = None):
        super().__init__()
        self.robot = robot
        self.cfg = cfg or EnvConfig()
        a, s = self.cfg.arena, self.cfg.sentinels
        if s.num_sentinels > s.max_sentinels:
            raise ValueError("num_sentinels cannot exceed max_sentinels")

        self.basic_ts = int(robot.getBasicTimeStep())
        self.control_ms = max(self.basic_ts, self.cfg.control_period_ms)
        self.dt = self.control_ms / 1000.0

        # --- devices -------------------------------------------------------
        self.left_motor = robot.getDevice("left wheel motor")
        self.right_motor = robot.getDevice("right wheel motor")
        for m in (self.left_motor, self.right_motor):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
        self.ir = [robot.getDevice(f"ps{i}") for i in range(NUM_IR_SENSORS)]
        for sensor in self.ir:
            sensor.enable(self.basic_ts)

        # --- supervisor handles -------------------------------------------
        self.self_node = robot.getSelf()
        self.trans_field = self.self_node.getField("translation")
        self.rot_field = self.self_node.getField("rotation")
        self.spawn_z = self.trans_field.getSFVec3f()[2]

        self._place_node(self._ensure_node("CHECKPOINT", pad_vrml("CHECKPOINT", a.checkpoint_radius, (0.1, 0.8, 0.2))), a.checkpoint, 0.0005)
        self._place_node(self._ensure_node("EXIT", pad_vrml("EXIT", a.exit_radius, (0.1, 0.3, 0.9))), a.exit, 0.0005)

        self.sentinels: List[WaypointSentinel] = []
        self.sentinel_nodes = []
        for i in range(s.num_sentinels):
            route = s.routes[i % len(s.routes)]
            self.sentinels.append(WaypointSentinel(route, s.speed))
            name = f"SENTINEL_{i}"
            self.sentinel_nodes.append(
                self._ensure_node(name, sentinel_vrml(name, s.detection_range, s.detection_half_angle))
            )
        # Hide sentinels that exist in the world file but are not active.
        for i in range(s.num_sentinels, s.max_sentinels):
            node = robot.getFromDef(f"SENTINEL_{i}")
            if node is not None:
                self._place_node(node, (0.0, 0.0), -5.0)

        # --- spaces --------------------------------------------------------
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        obs_dim = observation_size(s.max_sentinels)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # --- episode state -------------------------------------------------
        self.steps = 0
        self.checkpoint_reached = False
        self.prev_target_dist = 0.0

    # ------------------------------------------------------------------ #
    # Webots helpers
    # ------------------------------------------------------------------ #
    def _ensure_node(self, def_name: str, vrml: str):
        node = self.robot.getFromDef(def_name)
        if node is None:
            self.robot.getRoot().getField("children").importMFNodeFromString(-1, vrml)
            node = self.robot.getFromDef(def_name)
        return node

    @staticmethod
    def _place_node(node, xy: Vec2, z: float, heading: Optional[float] = None) -> None:
        node.getField("translation").setSFVec3f([float(xy[0]), float(xy[1]), float(z)])
        if heading is not None:
            node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, float(heading)])

    def _sim_step(self) -> None:
        if self.robot.step(self.control_ms) == -1:
            raise SimulationEnded("Webots simulation ended")

    def _robot_pose(self) -> Tuple[Vec2, float]:
        p = self.self_node.getPosition()
        r = self.self_node.getOrientation()  # 3x3 row-major
        return (p[0], p[1]), math.atan2(r[3], r[0])

    def _ir_values(self) -> List[float]:
        return [s.getValue() for s in self.ir]

    def _sync_sentinel_nodes(self) -> None:
        for sentinel, node in zip(self.sentinels, self.sentinel_nodes):
            self._place_node(node, sentinel.position, 0.0, sentinel.heading)

    # ------------------------------------------------------------------ #
    # Episode logic
    # ------------------------------------------------------------------ #
    @property
    def target(self) -> Vec2:
        return self.cfg.arena.exit if self.checkpoint_reached else self.cfg.arena.checkpoint

    def _spawn_pose(self) -> Tuple[Vec2, float]:
        a = self.cfg.arena
        if a.randomize_spawn:
            # TODO (final goal): sample a free spot far from checkpoint/exit/sentinels.
            h = a.half_size - 0.15
            xy = tuple(self.np_random.uniform(-h, h, size=2))
            return (float(xy[0]), float(xy[1])), float(self.np_random.uniform(-math.pi, math.pi))
        return a.spawn, a.spawn_heading

    def _observe(self) -> np.ndarray:
        xy, heading = self._robot_pose()
        return build_observation(
            self._ir_values(),
            xy,
            heading,
            self.target,
            self.checkpoint_reached,
            [(s.position, s.heading) for s in self.sentinels],
            self.cfg.sentinels.max_sentinels,
            self.cfg.arena.half_size,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        spawn_xy, spawn_heading = self._spawn_pose()
        self.trans_field.setSFVec3f([spawn_xy[0], spawn_xy[1], self.spawn_z])
        self.rot_field.setSFRotation([0.0, 0.0, 1.0, spawn_heading])
        self.self_node.resetPhysics()

        for sentinel in self.sentinels:
            sentinel.reset()
        self._sync_sentinel_nodes()

        self._sim_step()  # let Webots apply the new poses

        self.steps = 0
        self.checkpoint_reached = False
        xy, _ = self._robot_pose()
        self.prev_target_dist = distance(xy, self.target)
        return self._observe(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        vmax = self.cfg.max_wheel_speed
        self.left_motor.setVelocity(float(a[0]) * vmax)
        self.right_motor.setVelocity(float(a[1]) * vmax)

        for sentinel in self.sentinels:
            sentinel.step(self.dt)
        self._sync_sentinel_nodes()

        self._sim_step()
        self.steps += 1

        arena, s_cfg = self.cfg.arena, self.cfg.sentinels
        xy, _ = self._robot_pose()

        reached_checkpoint = False
        if not self.checkpoint_reached and distance(xy, arena.checkpoint) <= arena.checkpoint_radius:
            self.checkpoint_reached = True
            reached_checkpoint = True
            # Switch target: restart progress shaping from the new target distance.
            self.prev_target_dist = distance(xy, self.target)

        target_dist = distance(xy, self.target)
        reached_exit = self.checkpoint_reached and target_dist <= arena.exit_radius
        detected = any(
            in_fov_cone(s.position, s.heading, xy, s_cfg.detection_range, s_cfg.detection_half_angle)
            for s in self.sentinels
        )
        near_wall = max(self._ir_values()) / IR_MAX_VALUE > self.cfg.wall_ir_threshold

        events = StepEvents(
            prev_target_dist=self.prev_target_dist,
            target_dist=target_dist,
            reached_checkpoint=reached_checkpoint,
            reached_exit=reached_exit,
            detected=detected,
            near_wall=near_wall,
        )
        reward, terminated = compute_reward(events, self.cfg.rewards)
        self.prev_target_dist = target_dist
        truncated = (not terminated) and self.steps >= self.cfg.max_episode_steps

        info = {
            "is_success": bool(reached_exit and not detected),
            "detected": bool(detected),
            "checkpoint_reached": bool(self.checkpoint_reached),
            "steps": self.steps,
        }
        return self._observe(), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
