from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import mujoco
import jax
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation

from achql.brax.envs.manipulation.ur5e_envs import UR5eEnvs


"""
Reach: Translate end of arm along given velocity vector
- Observation space: 6-dim obs
- Action space:      4-dim, each element in [-1, 1], corresponding to target angles for joints 1, 2, 4, 6.

See _get_obs() and UR5eEnvs._convert_action() for details.
"""
class UR5eTranslate(PipelineEnv):

    def __init__(
            self,
            backend="mjx",
            healthy_angle_range=(-0.196349540849, 0.196349540849),
            healthy_reward=5.0,
            terminate_when_unhealthy=False,
            target_velocity_vec: jnp.array = jnp.array([1.0, 0.0, 0.0]),
            target_disp_vec: jnp.array = None,
            **kwargs
    ):
        self.healthy_reward = healthy_reward
        self.terminate_when_unhealthy = terminate_when_unhealthy
        self.healthy_angle_range = healthy_angle_range
        self.target_velocity_vec = target_velocity_vec
        self.target_disp_vec = target_disp_vec

        # links 1-6 are indices 0-5. The end-effector (eef) base is merged with
        # link 5, so we say link 5 index = eef index.
        self.eef_index = 5

        # Configure environment information (e.g. env name, noise scale, observation dimension, goal indices) and load XML
        self._set_environment_attributes()
        xml_path = self._get_xml_path()
        sys = mjcf.load(xml_path)

        # Configure backend
        sys = sys.tree_replace({
            "opt.timestep": 0.002,
            "opt.iterations": 6,
            "opt.ls_iterations": 12,
        })
        self.n_frames = 25
        kwargs["n_frames"] = kwargs.get("n_frames", self.n_frames)

        # Initialize brax PipelineEnv
        if backend != "mjx":
            raise Exception("Use the mjx backend for stability/reasonable speed.")
        super().__init__(sys=sys, backend=backend, **kwargs)

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

        # Initialize simulator state
        rng, subkey = jax.random.split(rng)
        q, qd = self._get_initial_state(subkey) # Injects noise to avoid overfitting/open loop control
        pipeline_state = self.pipeline_init(q, qd)
        timestep = 0.0

        # The default out vector of the EEF on the model is 0,0,1
        # initial_eef_rot = Rotation.from_quat(pipeline_state.x.rot[self.eef_index])
        # initial_eef_out_vector = initial_eef_rot.apply(jnp.array([0.0, 0.0, 1.0]))
        initial_eef_out_vector = jnp.array([0.0, 0.0, -1.0])
        eef_position = pipeline_state.x.pos[self.eef_index]

        # Fill info variable.
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        info = {
            "initial_eef_out_vector": initial_eef_out_vector,
            "seed": 0, # Seed is required, but fill it with a dummy value
            "timestep": 0.0,
            "postexplore_timestep": jax.random.uniform(subkey2) # Assumes timestep is normalized between 0 and 1
        }

        # Get components for state (observation, reward, metrics)
        obs = self._get_obs(pipeline_state, timestep)
        reward, done, zero = jnp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_healthy": zero,
            "x_position": eef_position[0],
            "y_position": eef_position[1],
            "z_position": eef_position[2],
            "x_velocity": zero,
            "y_velocity": zero,
            "z_velocity": zero,
        }
        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""

        # Run mujoco step
        pipeline_state0 = state.pipeline_state
        arm_angles = self._get_arm_angles(pipeline_state0)
        action = self._convert_action_to_actuator_input_joint_angle(action, arm_angles, delta_control=True)

        pipeline_state = self.pipeline_step(pipeline_state0, action)

        # Compute variables for state update, including observation and goal/reward
        timestep = state.info["timestep"] + 1 / self.episode_length
        obs = self._get_obs(pipeline_state, timestep)

        eef_position = pipeline_state.x.pos[self.eef_index]
        eef_velocity = (pipeline_state.x.pos[self.eef_index] - pipeline_state0.x.pos[self.eef_index]) / self.dt

        if self.target_velocity_vec is not None:
            forward_reward = jnp.dot(eef_velocity, self.target_velocity_vec)
        elif self.target_disp_vec is not None:
            forward_reward = jnp.dot(eef_position, self.target_disp_vec)
        else:
            raise NotImplementedError()

        u = state.info["initial_eef_out_vector"]
        eef_rot = Rotation.from_quat(pipeline_state.x.rot[self.eef_index])
        v = eef_out_vector = eef_rot.apply(jnp.array([0.0, 0.0, 1.0]))
        deviation_angle = jnp.arccos(jnp.dot(u, v) / (jnp.linalg.norm(u) * jnp.linalg.norm(v)))

        min_angle, max_angle = self.healthy_angle_range
        is_healthy = jnp.where(deviation_angle < min_angle, 0.0, 1.0)
        is_healthy = jnp.where(deviation_angle > max_angle, 0.0, is_healthy)
        if self.terminate_when_unhealthy:
            healthy_reward = self.healthy_reward
        else:
            healthy_reward = self.healthy_reward * is_healthy

        reward = forward_reward + healthy_reward
        done = 1.0 - is_healthy if self.terminate_when_unhealthy else 0.0
        info = {**state.info, "timestep": timestep}

        new_state = state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=info)
        new_state.metrics.update(
            reward_forward=forward_reward,
            reward_healthy=healthy_reward,
            x_position=eef_position[0],
            y_position=eef_position[1],
            z_position=eef_position[2],
            x_velocity=eef_velocity[0],
            y_velocity=eef_velocity[1],
            z_velocity=eef_velocity[2],
        )
        return new_state

    @property
    def action_size(self) -> int:
        return 5 # Override default (actuator count)

    def _convert_action_to_actuator_input_joint_angle(self, action: jax.Array, arm_angles: jax.Array, delta_control=False) -> jax.Array:
        """
        Converts the [-1, 1] actions to the corresponding target angle or gripper strength.
        We use the exact numbers for radians specified in the XML, even if they might be cleaner in terms of pi.

        We restrict rotation to approximately the front two octants, and further restrict wrist rotation, to
        reduce the space of unreasonable actions. Hence the action dimension for the arm joints is 4 instead of 6,
        though the action given to the simulator itself needs to be 6 for the arm, plus 1 for the robotiq input.

        delta_control: indicates whether or not to interpret the arm actions as targeting an offset from the current angle, or as
        targeting an absolute angle. Using delta control might improve convergence by reducing the effective action space at any timestep.
        """

        arm_action = jnp.array([action[0], action[1], action[2], action[3], action[4], 0.0000]) # Expand from 5-dim to 6-dim, and fill in fixed values for joint 5
        min_value = jnp.array([0.0000, -2.2000 - 1.57, 1.9000 - 1.57, -1.3830 - 1.57, -1.5700 - 1.57, 0.0000])
        max_value = jnp.array([3.1415, -2.2000 + 1.57, 1.9000 + 1.57, -1.3830 + 1.57, -1.5700 + 1.57, 0.0000])

        # If f(x) = offset + x * multiplier, then this offset and multiplier yield f(-1) = min_value, f(1) = max_value.
        offset = (min_value + max_value) / 2
        multiplier = (max_value - min_value) / 2

        # Retrieve absolute angle target in [-1, 1] space from delta actions in [-1, 1]
        if delta_control:
            normalized_arm_angles = jnp.where(multiplier > 0, (arm_angles - offset) / multiplier, 0) # Convert arm angles back to [-1, 1] space
            delta_range = 0.25 # If this number is 0.25, an action of +/- 1 targets an angle 25% of the max range away from the current angle.
            arm_action = normalized_arm_angles + arm_action * delta_range
            arm_action = jnp.clip(arm_action, -1, 1)

        # Rescale back to absolute angle space in radians
        arm_action = offset + arm_action * multiplier

        # Gripper control
        # Binary open-closedness: if positive, set to actuator value 0 (totally closed); if negative, set to actuator value 255 (totally open)
        # The UR5e Gripper (Robotiq-85), has a single control input
        if self.env_name not in ("ur5e_reach", "ur5e_reach_shelf"):
            gripper_action = jnp.where(action[-1] > 0, jnp.array([0], dtype=float), jnp.array([255], dtype=float))
            converted_action = jnp.concatenate([arm_action] + [gripper_action])
        else:
            converted_action = arm_action

        return converted_action

    # Methods to be overridden by specific environments
    def _get_xml_path(self):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'assets', "ur5e_translate.xml")

    def _set_environment_attributes(self):
        """
        Attribute descriptions:
        - Episode lengths are merely a recommendation + used for scaling timestep
        - Goal indices are for data collection/goal-conditioning; completion goal indices are for reward/goal completion checking
        - State_dim is for the observation dimension WITHOUT the goal appended; brax's env.observation_size gives the dimension WITH goal appended
        - Noise scale is for the range of uniform-distribution noise
            - Arm: perturb all joints (in radians) and gripper position (in meters)
            - Cube: perturb x, y of starting cube location
            - Goal: perturb x, y of cube destination or x, y, z of reach location
        """
        self.env_name = "ur5e_reach"
        self.episode_length = 100

        self.pos_indices = jnp.array([6, 7, 8]) # End-effector position
        self.goal_indices = jnp.array([12, 13, 14]) # End-effector position
        self.completion_goal_indices = jnp.array([6, 7, 8]) # Identical
        self.state_dim = 12
        self.goal_dist = 0.1

        self.arm_noise_scale = 0.0
        self.goal_noise_scale = 0.2

    def _get_initial_state(self, rng):
        arm_q_default = jnp.array([1.5708, -2.2000, 1.9000, -1.3830, -1.5700,  0.0000]) # Start closer to the relevant area
        arm_q = arm_q_default + self.arm_noise_scale * jax.random.uniform(rng, [self.sys.q_size()], minval=-1)

        q = arm_q
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd

    def _get_obs(self, pipeline_state: base.State, timestep) -> jax.Array:
        """
        Observation space (13-dim)
         - q_subset (6-dim): joint angles
         - End of arm (6-dim): position and velocity
        Note q is 14-dim: 7-dim cube position/angle, 6-dim joint angles

        Goal space (3-dim): position of end of arm
        """

        joint_angles = pipeline_state.q
        eef_index = 5
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]

        return jnp.concatenate([joint_angles] + [eef_x_pos] + [eef_xd_vel])

    def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
        return pipeline_state.q
