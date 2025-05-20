"""Hopper environment with a safety constraint on velocity."""

from acql.brax.safety_gymnasium_envs.safety_hopper_velocity_v0 import (
    SafetyHopperVelocityEnv as HopperEnv,
)


class SafetyHopperVelocityEnv(HopperEnv):
    """Hopper environment with a safety constraint on velocity."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._velocity_threshold = 0.7402
