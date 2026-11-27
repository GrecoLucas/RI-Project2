import math

import numpy as np
import pytest

from config import RewardConfig
from logic import (
    StepEvents,
    build_observation,
    compute_reward,
    in_fov_cone,
    observation_size,
    to_robot_frame,
    wrap_angle,
)
from sentinels import WaypointSentinel


def test_wrap_angle():
    assert wrap_angle(0.0) == pytest.approx(0.0)
    assert wrap_angle(3 * math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(-math.pi / 2) == pytest.approx(-math.pi / 2)


def test_to_robot_frame():
    # Robot at origin facing +y: a point at +y is straight ahead, a point at -x is to the left.
    fx, fy = to_robot_frame((0, 0), math.pi / 2, (0, 1))
    assert (fx, fy) == pytest.approx((1, 0))
    lx, ly = to_robot_frame((0, 0), math.pi / 2, (-1, 0))
    assert (lx, ly) == pytest.approx((0, 1))


@pytest.mark.parametrize(
    "target, expected",
    [
        ((0.3, 0.0), True),  # straight ahead, in range
        ((0.5, 0.0), False),  # too far
        ((0.2, 0.2), False),  # 45 deg, outside a 30 deg half-angle
        ((0.3, 0.1), True),  # ~18 deg, inside
        ((-0.2, 0.0), False),  # behind
    ],
)
def test_in_fov_cone(target, expected):
    assert in_fov_cone((0, 0), 0.0, target, 0.4, math.radians(30)) is expected


def test_sentinel_moves_and_loops():
    s = WaypointSentinel([(0, 0), (1, 0), (1, 1)], speed=1.0)
    s.step(0.5)
    assert s.position == pytest.approx((0.5, 0.0))
    assert s.heading == pytest.approx(0.0)
    s.step(1.0)  # passes (1, 0), heads up
    assert s.position == pytest.approx((1.0, 0.5))
    assert s.heading == pytest.approx(math.pi / 2)
    s.step(0.5 + math.sqrt(2))  # reaches (1,1) then back to (0,0)
    assert s.position == pytest.approx((0.0, 0.0), abs=1e-9)
    s.reset()
    assert s.position == (0.0, 0.0)


def test_sentinel_needs_two_waypoints():
    with pytest.raises(ValueError):
        WaypointSentinel([(0, 0)], 1.0)


def test_observation_shape_and_padding():
    obs = build_observation(
        ir_values=[0] * 8,
        robot_xy=(0, 0),
        robot_heading=0.0,
        target_xy=(1, 0),
        checkpoint_reached=False,
        sentinels=[((0.5, 0.0), math.pi)],
        max_sentinels=3,
        arena_half_size=1.0,
    )
    assert obs.shape == (observation_size(3),)
    assert obs.dtype == np.float32
    slot0 = obs[16:21]
    assert slot0[0] == 1.0  # present
    assert np.all(obs[21:] == 0.0)  # padded slots


def test_reward_terms():
    cfg = RewardConfig()
    r, term = compute_reward(StepEvents(1.0, 0.9), cfg)
    assert not term
    assert r == pytest.approx(cfg.step_penalty + cfg.progress_scale * 0.1)

    r, term = compute_reward(StepEvents(0.5, 0.5, detected=True), cfg)
    assert term and r == pytest.approx(cfg.step_penalty + cfg.detected_penalty)

    r, term = compute_reward(StepEvents(0.1, 0.1, reached_exit=True), cfg)
    assert term and r == pytest.approx(cfg.step_penalty + cfg.exit_bonus)

    r, term = compute_reward(StepEvents(0.1, 0.1, reached_checkpoint=True, near_wall=True), cfg)
    assert not term
    assert r == pytest.approx(cfg.step_penalty + cfg.checkpoint_bonus + cfg.wall_penalty)
