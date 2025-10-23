'''import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.envs.registration import register

# Register the custom environment
register(
    id='CustomCartPoleEnv-v0',
    entry_point='gym_wrapper:CustomCartPoleEnv',
    max_episode_steps=500,
)

class CustomCartPoleEnv(gym.Env):
    # This class is a placeholder for your environment and should contain the
    # physics simulation, as you have it implemented. The important thing is
    # that its step method returns the 5-element tuple.
    def __init__(self):
        super(CustomCartPoleEnv, self).__init__()
        # ... (rest of your __init__ code)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array([-2.4, -np.inf, -0.418, -np.inf]),
                                                high=np.array([2.4, np.inf, 0.418, np.inf]),
                                                dtype=np.float32)
        # ... (rest of your __init__ code)
        self.state = None
        self.step_count = 0

    def step(self, action):
        # ... (your step logic here)
        # This function must return the 5-element gymnasium tuple.
        next_state = np.array(self.state, dtype=np.float32)
        reward = 1.0 if not terminated else 0.0
        return next_state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # ... (your reset logic here)
        state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state = state
        self.step_count = 0
        return np.array(self.state, dtype=np.float32), {}


class GymWrapper:
    def __init__(self, env):
        self.env = env
        # Set the observation space of the wrapper to match the filtered state
        # The shape should be (2,) for pole angle and angular velocity
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = self.env.action_space
        
    def _filter_state(self, state):
        # Keep only the pole angle and angular velocity
        return np.array(state[2:4], dtype=np.float32)

    def reset(self):
        state, info = self.env.reset()
        return self._filter_state(state), info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        filtered_state = self._filter_state(next_state)
        return filtered_state, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()'''

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.envs.registration import register

# Register the custom environment
register(
    id='CustomCartPoleEnv-v0',
    entry_point='gym_wrapper:CustomCartPoleEnv',
    max_episode_steps=500,
)

class CustomCartPoleEnv(gym.Env):
    def __init__(self):
        super(CustomCartPoleEnv, self).__init__()
        
        # Define action space (continuous motor speed, e.g., from -1.0 to 1.0)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Define observation space
        self.observation_space = gym.spaces.Box(low=np.array([-2.4, -np.inf, -0.418, -np.inf]),
                                                high=np.array([2.4, np.inf, 0.418, np.inf]),
                                                dtype=np.float32)
        
        # Environment constants (copied from your original code)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        self.state = None
        self.steps_beyond_done = None
        self.step_count = 0

    def step(self, action):
        motor_speed = action
        force = self.force_mag * motor_speed
        
        x, x_dot, theta, theta_dot = self.state
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)
        
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        self.step_count += 1
        truncated = self.step_count >= 500

        reward = 1.0 if not terminated else 0.0
        
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.step_count = 0
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        # Your render logic here
        pass

    def close(self):
        # Your close logic here
        pass

# The wrapper class
class GymWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = self._filter_state(self.env.observation_space.sample()).shape
        self.action_space = self.env.action_space
        
    def _filter_state(self, state):
        # Keep only the pole angle and angular velocity
        return np.array(state[2:4], dtype=np.float32)

    def reset(self):
        state, info = self.env.reset()
        return self._filter_state(state), info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        filtered_state = self._filter_state(next_state)
        return filtered_state, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()