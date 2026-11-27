"""Run SentinelEscapeEnv against a tiny fake Webots supervisor.

This does not replace testing in Webots, but it validates the Gymnasium API,
reset/step bookkeeping, checkpoint -> exit logic and detection without
needing Webots installed.
"""

import math

import numpy as np
import pytest

from config import EnvConfig
from env import SentinelEscapeEnv

WHEEL_RADIUS = 0.0205
AXLE = 0.052


class FakeField:
    def __init__(self, value):
        self.value = list(value)

    def getSFVec3f(self):
        return list(self.value)

    def setSFVec3f(self, v):
        self.value = list(v)

    def setSFRotation(self, v):
        self.value = list(v)


class FakeNode:
    def __init__(self, translation=(0, 0, 0)):
        self.fields = {"translation": FakeField(translation), "rotation": FakeField((0, 0, 1, 0))}

    def getField(self, name):
        return self.fields[name]

    def getPosition(self):
        return self.fields["translation"].getSFVec3f()

    def getOrientation(self):
        a = self.fields["rotation"].value[3]
        c, s = math.cos(a), math.sin(a)
        return [c, -s, 0, s, c, 0, 0, 0, 1]

    def resetPhysics(self):
        pass


class FakeMotor:
    def __init__(self):
        self.velocity = 0.0

    def setPosition(self, p):
        pass

    def setVelocity(self, v):
        self.velocity = v


class FakeSensor:
    def enable(self, ts):
        pass

    def getValue(self):
        return 0.0


class FakeChildren:
    def __init__(self, sup):
        self.sup = sup

    def importMFNodeFromString(self, pos, vrml):
        name = vrml.split("DEF ")[1].split()[0]
        self.sup.defs[name] = FakeNode()


class FakeRoot:
    def __init__(self, sup):
        self.children = FakeChildren(sup)

    def getField(self, name):
        return self.children


class FakeSupervisor:
    """Differential-drive kinematics, enough to exercise the env."""

    def __init__(self):
        self.agent = FakeNode((-0.8, -0.8, 0.0))
        self.defs = {}
        self.devices = {"left wheel motor": FakeMotor(), "right wheel motor": FakeMotor()}
        for i in range(8):
            self.devices[f"ps{i}"] = FakeSensor()
        self.root = FakeRoot(self)

    def getBasicTimeStep(self):
        return 32

    def getSelf(self):
        return self.agent

    def getFromDef(self, name):
        return self.defs.get(name)

    def getRoot(self):
        return self.root

    def getDevice(self, name):
        return self.devices[name]

    def step(self, ms):
        dt = ms / 1000.0
        vl = self.devices["left wheel motor"].velocity * WHEEL_RADIUS
        vr = self.devices["right wheel motor"].velocity * WHEEL_RADIUS
        x, y, z = self.agent.fields["translation"].value
        a = self.agent.fields["rotation"].value[3]
        v, w = (vl + vr) / 2, (vr - vl) / AXLE
        a += w * dt
        x = float(np.clip(x + v * math.cos(a) * dt, -0.97, 0.97))
        y = float(np.clip(y + v * math.sin(a) * dt, -0.97, 0.97))
        self.agent.fields["translation"].value = [x, y, z]
        self.agent.fields["rotation"].value = [0, 0, 1, a]
        return 0


def make_env(num_sentinels=1):
    cfg = EnvConfig()
    cfg.sentinels.num_sentinels = num_sentinels
    return SentinelEscapeEnv(FakeSupervisor(), cfg)


def test_gymnasium_api():
    from stable_baselines3.common.env_checker import check_env

    check_env(make_env(), warn=True)


def test_reset_spawns_sentinels_and_obs_shape():
    env = make_env(num_sentinels=2)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert "SENTINEL_0" in env.robot.defs and "SENTINEL_1" in env.robot.defs
    assert np.all(np.isfinite(obs))


def drive_to(env, goal, max_steps=2000):
    """Simple go-to-goal controller, returns the last step's output."""
    out = None
    for _ in range(max_steps):
        x, y, _ = env.robot.agent.fields["translation"].value
        a = env.robot.agent.fields["rotation"].value[3]
        err = math.atan2(goal[1] - y, goal[0] - x) - a
        err = (err + math.pi) % (2 * math.pi) - math.pi
        turn = float(np.clip(2.0 * err, -1, 1))
        fwd = 0.6 if abs(err) < 0.5 else 0.0
        out = env.step([fwd - turn, fwd + turn])
        if out[2] or out[3] or env.checkpoint_reached and goal == env.cfg.arena.checkpoint:
            return out
    return out


def test_checkpoint_then_exit_without_sentinels():
    env = make_env(num_sentinels=0)
    env.reset()
    out = drive_to(env, env.cfg.arena.checkpoint)
    assert env.checkpoint_reached
    out = drive_to(env, env.cfg.arena.exit)
    obs, reward, terminated, truncated, info = out
    assert terminated and info["is_success"]
    assert reward > env.cfg.rewards.exit_bonus / 2


def test_exit_without_checkpoint_does_not_finish():
    env = make_env(num_sentinels=0)
    env.reset()
    for _ in range(300):
        _, _, terminated, _, _ = env.step(env.action_space.sample() * 0)
        assert not terminated
    assert not env.checkpoint_reached


def test_detection_terminates_episode():
    env = make_env(num_sentinels=1)
    env.reset()
    # Teleport the robot right in front of the sentinel.
    s = env.sentinels[0]
    ahead = (s.position[0] + 0.15 * math.cos(s.heading), s.position[1] + 0.15 * math.sin(s.heading))
    env.robot.agent.fields["translation"].value = [ahead[0], ahead[1], 0.0]
    _, reward, terminated, _, info = env.step([0.0, 0.0])
    assert terminated and info["detected"] and not info["is_success"]
    assert reward < 0


@pytest.mark.slow
def test_short_ppo_run():
    from stable_baselines3 import PPO

    model = PPO("MlpPolicy", make_env(), n_steps=256, batch_size=64, device="cpu", verbose=0)
    model.learn(total_timesteps=512)
