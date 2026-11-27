"""Sentinel behaviours.

For the intermediate goals a sentinel simply patrols a fixed, looping list of
waypoints at constant speed (kinematic motion, no physics). Future behaviours
(random routes, chasing, learned policies, "personalities") should implement
the same small interface: ``reset()``, ``step(dt, ...)``, ``position``,
``heading``.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Vec2 = Tuple[float, float]


class WaypointSentinel:
    """Moves along a closed polyline of waypoints at constant speed."""

    def __init__(self, waypoints: Sequence[Vec2], speed: float):
        if len(waypoints) < 2:
            raise ValueError("A patrol route needs at least two waypoints")
        self.waypoints: List[Vec2] = [tuple(map(float, w)) for w in waypoints]
        self.speed = float(speed)
        self.reset()

    def reset(self) -> None:
        self.position: Vec2 = self.waypoints[0]
        self._next = 1
        self.heading = self._bearing_to(self.waypoints[1])

    def _bearing_to(self, p: Vec2) -> float:
        return math.atan2(p[1] - self.position[1], p[0] - self.position[0])

    def step(self, dt: float) -> None:
        """Advance ``speed * dt`` metres along the route (may pass several waypoints)."""
        remaining = self.speed * dt
        while remaining > 1e-12:
            target = self.waypoints[self._next]
            dx, dy = target[0] - self.position[0], target[1] - self.position[1]
            seg = math.hypot(dx, dy)
            if seg <= remaining:
                self.position = target
                remaining -= seg
                self._next = (self._next + 1) % len(self.waypoints)
                self.heading = self._bearing_to(self.waypoints[self._next])
            else:
                self.position = (
                    self.position[0] + dx / seg * remaining,
                    self.position[1] + dy / seg * remaining,
                )
                self.heading = math.atan2(dy, dx)
                remaining = 0.0
