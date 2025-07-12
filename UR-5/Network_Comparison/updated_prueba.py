import gymnasium as gym
from gymnasium.core import ObsType, ActType, SupportsFloat, RenderFrame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import mujoco
import math
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Tuple, Union, List


from mujoco_env import MujocoEnv

class NewUR5ReachEnv(MujocoEnv):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple",],
    }

    def __init__(self, **kwargs):

        self.render_mode = kwargs.get("render_mode", None)

        default_camera_config = {
            "distance": 3.0,
            "elevation": -25.0,
            "azimuth": 170.0,
            "lookat": [0.0, 0.0, 0.0],
        }


        screen_width = screen_height = 800

        super().__init__(
            model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ur5_reach", "scene.xml"),
            frame_skip=15,
            observation_space=None,
            default_camera_config=default_camera_config,
            width=screen_width,
            height=screen_height,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.ee_link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")

        self.observation_space = gym.spaces.Box(
            low=np.array([-1]*6 + [-2*math.pi]*6, dtype=np.float32),
            high=np.array([+1]*6 + [+2*math.pi]*6, dtype=np.float32),
            shape=(12,)
        )
        action_max = 2 * math.pi / 10
        self.action_space = gym.spaces.Box(low=-action_max, high=+action_max, shape=(6,))

        self.steps = 0
        self.max_steps = 100

    def reset_model(self) -> np.ndarray:
        self.steps = 0
        mujoco.mj_resetData(self.model, self.data)
        
        self.target_pos = np.random.uniform(-0.5, 0.5, size=3).astype(np.float32)
        self.target_pos[2] = 0.7

        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")

        self.data.site_xpos[self.target_site_id] = self.target_pos

        return self.get_observation()

    def wait_until_stable(self, sim_steps=500) -> bool:
        def _get_eff_obs() -> np.ndarray:
            return np.concatenate((self.data.xpos[self.ee_link_id], self.data.qpos[:6]), axis=0)
        current_obs = _get_eff_obs()
        for _ in range(sim_steps):
            mujoco.mj_step(self.model, self.data)
            self.data.site_xpos[self.target_site_id] = self.target_pos
            if self.render_mode == "human":
                self.render()
            new_obs = _get_eff_obs()
            if np.sum(np.abs(current_obs - new_obs)) < 5e-3:
                return True
            current_obs = new_obs.copy()
        return False

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action: {action}"
        self.steps += 1

        self.data.ctrl[:] += action
        self.data.ctrl[:] = np.clip(self.data.ctrl, -2 * math.pi, 2 * math.pi)
        self.data.site_xpos[self.target_site_id] = self.target_pos
        
        self.wait_until_stable() 

        obs = self.get_observation()
        
        ee_pos = obs[3:6]
        distance = np.linalg.norm(self.target_pos - ee_pos)
        reward = -distance
        terminated = distance < 0.05
        if terminated:
            reward += 10
        truncated = self.steps >= self.max_steps
        info = {}
        if terminated or truncated:
            info["episode"] = {"l": self.steps, "r": reward}
        return obs, reward, terminated, truncated, info

    def get_observation(self) -> np.ndarray:
        ee_pos = self.data.xpos[self.ee_link_id]
        joint_pos = self.data.qpos[:6]
        return np.concatenate((self.target_pos, ee_pos, joint_pos), dtype=np.float32)

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.ee_link_id], self.data.xquat[self.ee_link_id]

# =====================================
# 1. Data collection in Reacher-v5
# =====================================
def collect_data(agent, n_episodes=100, max_steps=500):
    """
    Steps through the Reacher-v5 environment using a random policy and
    records transitions: (state, action, next_state).
    """
    env = NewUR5ReachEnv()  # replace with your Reacher-v5 env constructor
    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0

        for t in range(max_steps):
            steps += 1
            if done:
                break

            action, _ = agent.predict(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)

            state = next_state

        # print(f"Episode {ep}: steps = {steps}, total reward = {total_reward}")

    env.close()
    return (
        np.array(states),
        np.array(actions),
        np.array(next_states),
        np.array(rewards),
        np.array(dones),
    )


# =====================================
# 2. State prediction network with PyTorch
# =====================================
class StatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        The network takes as input the current state (dim=state_dim)
        and the action in one-hot format (dim=action_dim),
        and predicts the next state (dim=state_dim).
        """
        super(StatePredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim - 3, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            # nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, state_dim - 9)
        )

    def forward(self, state, action_batch):
        # Concatenate representations
        next_angles = (state[:, 6:] + action_batch).clamp(
            min=-2 * math.pi, max=2 * math.pi
        )
        next_joint_angles = next_angles / (2 * math.pi)
        x = torch.cat([state[:, 3:6], next_joint_angles], dim=1)

        # Residual connection: current state plus predicted delta
        delta = self.fc(x)
        next_state = state[:, 3:6] + delta

        return torch.cat([state[:, :3], next_state, next_angles], dim=1)


def prepare_dataloader(states, actions, next_states, rewards, dones, batch_size=32):
    """
    Prepares a DataLoader from the collected data.
    """
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    dones_tensor = torch.tensor(dones, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(
        states_tensor,
        actions_tensor,
        next_states_tensor,
        rewards_tensor,
        dones_tensor
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_predictor(model, dataloader, epochs=20, lr=1e-3, verbose=1):
    """
    Trains the network using MSELoss to minimize the difference between
    the true next state and the predicted one.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()

        for batch in dataloader:
            state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch

            optimizer.zero_grad()
            pred_next_state = model(state_batch, action_batch)
            loss = criterion(pred_next_state, next_state_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * state_batch.size(0)

        epoch_loss /= len(dataloader.dataset)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# ====================================
# 3. Reward Prediction Network
# ====================================
class RewardPredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        The network takes as input the current state (dim=state_dim),
        the action in one-hot format (dim=action_dim),
        and the next state (dim=state_dim),
        and predicts a scalar reward (difference between two logits).
        """
        super(RewardPredictor, self).__init__()
        # Primary branch
        self.fc = nn.Sequential(
            nn.Linear(state_dim - 6, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Secondary branch
        self.fc_special = nn.Sequential(
            nn.Linear(state_dim - 6, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, next_state):
        """
        Computes two logits from the negative of the first 6 dims of next_state,
        then returns their difference as the reward.
        """
        neg_obs = -next_state[:, :6]
        logits = self.fc(neg_obs)
        special = self.fc_special(neg_obs)
        reward = special - logits
        return reward.squeeze(1)


def train_reward_predictor(model, dataloader, epochs=20, lr=1e-3, verbose=1):
    """
    Trains the reward predictor using SmoothL1Loss to minimize the difference
    between the predicted reward and the true reward label.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for batch in dataloader:
            _, _, next_state_batch, reward_batch, _ = batch
            optimizer.zero_grad()
            pred_reward = model(next_state_batch)
            loss = criterion(pred_reward, reward_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * next_state_batch.size(0)
        
        epoch_loss /= len(dataloader.dataset)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")


# ====================================
# 4. Termination Prediction Network
# ====================================
class TerminationPredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        The network takes as input the next state (dim=state_dim)
        and predicts the termination probability (scalar between 0 and 1).
        """
        super(TerminationPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim - 6, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, next_state):
        """
        Applies a sigmoid to the network output on the first 6 dims of next_state
        to obtain a termination probability.
        """
        logits = self.fc(next_state[:, :6])
        return torch.sigmoid(logits)


def train_termination_predictor(model, dataloader, epochs=20, lr=1e-3, verbose=1):
    """
    Trains the termination predictor using BCELoss to minimize the difference
    between the predicted probability and the true termination label (0 or 1).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for batch in dataloader:
            _, _, next_state_batch, _, termination_batch = batch
            optimizer.zero_grad()
            pred_term = model(next_state_batch)
            loss = criterion(pred_term, termination_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * next_state_batch.size(0)
        
        epoch_loss /= len(dataloader.dataset)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# =====================================
#   Unified World Model Network
# =====================================
class OneNetworkWM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        The network takes as input the current state (dim=state_dim)
        and the action vector (dim=action_dim), and predicts:
            - The next state: current state + estimated delta.
            - The step reward: a scalar value.
            - The termination probability: a scalar between 0 and 1.
        """
        super(OneNetworkWM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared trunk processing the concatenated state+action
        self.shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.1)
        )
        
        # Head for predicting the state delta (to compute next state)
        self.state_head = nn.Linear(hidden_dim // 2, state_dim)
        # Head for predicting the reward
        self.reward_head = nn.Linear(hidden_dim // 2, 1)
        # Head for predicting the termination probability
        self.termination_head = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, state, action):
        """
        state: Tensor of shape (batch_size, state_dim)
        action: Tensor of shape (batch_size, action_dim)
        Returns: next_state, reward, termination_prob
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        features = self.shared(x)
        
        # Predict state delta and add to current state
        delta_state = self.state_head(features)
        next_state = state + delta_state
    
        # Predict reward scalar
        reward = self.reward_head(features).squeeze(1)
        # Predict termination probability
        termination = torch.sigmoid(self.termination_head(features))
        
        return next_state, reward, termination


def train_one_net_wm_model(model, dataloader, epochs=20, lr=1e-3, device='cpu'):
    """
    Trains the unified world-model network that predicts:
      - Next state (current state + delta).
      - Step reward.
      - Termination probability.

    Uses:
      - MSELoss for state and reward predictions.
      - BCELoss for the termination flag.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    termination_criterion = nn.BCELoss()
    
    model.to(device)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        for batch in dataloader:
            # Expect each batch: state, action, next_state, reward, done_flag
            state_batch, action_batch, next_state_batch, reward_batch, termination_batch = batch
            
            # Move data to the target device
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            next_state_batch = next_state_batch.to(device)
            reward_batch = reward_batch.to(device)
            termination_batch = termination_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through the network
            pred_next_state, pred_reward, pred_termination = model(state_batch, action_batch)
            
            # Compute each loss term
            loss_state = criterion(pred_next_state, next_state_batch)
            loss_reward = criterion(pred_reward, reward_batch)
            loss_termination = termination_criterion(pred_termination, termination_batch)
            
            # Sum losses (weights can be adjusted if needed)
            loss = loss_state + loss_reward + loss_termination
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * state_batch.size(0)
        
        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# =====================================
# 4. Define the world-model environment using the predictor networks
# =====================================
class WorldModelReacherEnv(gym.Env):
    """
    A simulated environment (world model) that uses the predictor networks
    to evolve the state. Reward and termination logic follow the same
    rules as in CartPole-v1.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        state_predictor,
        reward_predictor,
        end_predictor,
        state_dim=12,
        action_dim=6
    ):
        super(WorldModelReacherEnv, self).__init__()
        self.state_predictor = state_predictor
        self.reward_predictor = reward_predictor
        self.end_predictor = end_predictor
        self.state_dim = state_dim

        # Load MuJoCo model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "ur5_reach", "scene.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.ee_link_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link"
        )

        # Define action and observation spaces
        action_max = 2 * math.pi / 10
        self.action_space = gym.spaces.Box(
            low=-action_max, high=+action_max, shape=(action_dim,)
        )
        obs_low = np.array([-1.0] * 6 + [-2*math.pi] * 6, dtype=np.float32)
        obs_high = np.array([+1.0] * 6 + [+2*math.pi] * 6, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, shape=(state_dim,)
        )

        self.total_reward = 0.0
        self.steps = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        """
        Resets the environment. Samples a new random target position
        and returns the initial observation.
        """
        self.steps = 0
        self.total_reward = 0.0

        # Sample a random target: x,y ∈ [-0.5, 0.5], z = 0.7
        self.target_pos = np.random.uniform(-0.5, 0.5, size=3)
        self.target_pos[2] = 0.7

        self.state = self.get_observation()
        return self.state, {}

    def get_observation(self) -> np.ndarray:
        """
        Retrieves the end-effector position and the joint angles,
        then concatenates: [target_pos (3), ee_pos (3), joint_angles (6)].
        """
        self.ee_pos = self.data.xpos[self.ee_link_id]
        joint_pos = self.data.qpos[:6]
        return np.concatenate(
            (self.target_pos, self.ee_pos, joint_pos),
            dtype=np.float32
        )

    def wait_until_stable(self, sim_steps=500) -> bool:
        """
        Advances the MuJoCo simulation until the model settles
        or until sim_steps have been executed.
        """
        def _read_obs() -> np.ndarray:
            return np.concatenate(
                (self.data.xpos[self.ee_link_id], self.data.qpos[:6]),
                axis=0
            )

        prev_obs = _read_obs()
        for _ in range(sim_steps):
            mujoco.mj_step(self.model, self.data)
            if self.render_mode == "human":
                self.render()
            new_obs = _read_obs()
            if np.sum(np.abs(prev_obs - new_obs)) < 5e-3:
                return True
            prev_obs = new_obs.copy()

        return False

    def step(self, action):
        """
        Applies an action in the simulated model, uses the predictor networks
        to estimate next state/reward/termination, and occasionally injects
        a real observation to reduce drift.
        """
        self.steps += 1

        # Apply and clip control
        self.data.ctrl[:] += action
        self.data.ctrl[:] = np.clip(self.data.ctrl, -2*math.pi, 2*math.pi)

        self.wait_until_stable()

        # Prepare tensors for prediction
        state_tensor = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

        self.state_predictor.eval()
        self.reward_predictor.eval()
        self.end_predictor.eval()
        with torch.no_grad():
            # Predict next state
            next_state_tensor = self.state_predictor(state_tensor, action_tensor)
            next_state = next_state_tensor.squeeze(0).numpy()

            # Inject true simulator observation every 5 steps
            if self.steps % 5 == 0:
                next_state = self.get_observation()
                next_state_tensor = torch.tensor(
                    next_state, dtype=torch.float32
                ).unsqueeze(0)

            # Predict reward
            reward = self.reward_predictor(
                next_state_tensor
            ).squeeze(0).numpy()

            # Predict termination flag
            terminated = bool(
                (self.end_predictor(next_state_tensor) >= 0.5).item()
            )

        self.state = next_state
        self.total_reward += reward
        truncated = (self.steps >= self.max_steps)

        info = {}
        if terminated or truncated:
            # Include episode length and cumulative reward
            info["episode"] = {"l": self.steps, "r": self.total_reward}

        return self.state, reward, terminated, truncated, info

#
#   New WM
#
class NewWorldModelReacherEnv(gym.Env):
    """
    A simulated (world-model) environment that uses a single neural network
    predictor to evolve the state. Reward and termination logic follow the
    same rules as in CartPole-v1.
    """
    metadata = {"render_modes": []}

    def __init__(self, one_net_predictor, state_dim=12, action_dim=6):
        super(NewWorldModelReacherEnv, self).__init__()
        self.one_net_predictor = one_net_predictor
        self.state_dim = state_dim

        # Load the MuJoCo model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "ur5_reach", "scene.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.ee_link_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link"
        )

        # Action and observation spaces
        action_max = 2 * math.pi / 10
        self.action_space = gym.spaces.Box(
            low=-action_max, high=action_max, shape=(action_dim,)
        )
        obs_low = np.array([-1.0] * 6 + [-2*math.pi] * 6, dtype=np.float32)
        obs_high = np.array([+1.0] * 6 + [+2*math.pi] * 6, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, shape=(state_dim,)
        )

        self.total_reward = 0.0
        self.steps = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        """
        Resets the simulation state, samples a new target position, and
        returns the initial observation.
        """
        self.steps = 0
        self.total_reward = 0.0

        # Random target in x,y ∈ [-0.5,0.5], z = 0.7
        self.target_pos = np.random.uniform(-0.5, 0.5, size=3)
        self.target_pos[2] = 0.7

        self.state = self.get_observation()
        return self.state, {}

    def get_observation(self) -> np.ndarray:
        """
        Reads the end-effector position and joint angles, then concatenates:
        [target_pos (3), ee_pos (3), joint_angles (6)] → 12-dim observation.
        """
        self.ee_pos = self.data.xpos[self.ee_link_id]
        joint_pos = self.data.qpos[:6]
        return np.concatenate((self.target_pos, self.ee_pos, joint_pos), dtype=np.float32)

    def wait_until_stable(self, sim_steps=500) -> bool:
        """
        Advances the simulator until the model settles (low change in observation),
        or until sim_steps have been executed.
        """
        def _read_obs() -> np.ndarray:
            return np.concatenate((self.data.xpos[self.ee_link_id], self.data.qpos[:6]), axis=0)

        prev = _read_obs()
        for _ in range(sim_steps):
            mujoco.mj_step(self.model, self.data)
            if self.render_mode == "human":
                self.render()
            curr = _read_obs()
            if np.sum(np.abs(prev - curr)) < 5e-3:
                return True
            prev = curr.copy()
        return False

    def step(self, action):
        """
        Applies the action in the simulated model, uses the predictor network
        to estimate next_state, reward, and done flags, and occasionally
        injects a real observation to limit drift.
        """
        self.steps += 1

        # Apply control and clip
        self.data.ctrl[:] += action
        self.data.ctrl[:] = np.clip(self.data.ctrl, -2*math.pi, 2*math.pi)

        self.wait_until_stable()

        # Wrap current state and action as tensors
        state_t = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

        self.one_net_predictor.eval()
        with torch.no_grad():
            # Predict next state, reward, and termination probability
            next_state_t, reward_t, term_t = self.one_net_predictor(state_t, action_t)

            next_state = next_state_t.squeeze(0).numpy()
            reward = reward_t.squeeze(0).numpy()
            terminated = bool((term_t >= 0.5).item())

            # Every 5 steps, inject the true simulator observation
            if self.steps % 5 == 0:
                next_state = self.get_observation()
                next_state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        self.state = next_state
        self.total_reward += reward_t
        truncated = (self.steps >= self.max_steps)

        info = {}
        if terminated or truncated:
            # Include episode length and cumulative reward
            info["episode"] = {"l": self.steps, "r": self.total_reward}

        return self.state, reward, terminated, truncated, info


# =====================================
# 6. Testing the trained policy on UR5
# =====================================
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeInfoCallback(BaseCallback):
    """
    Callback that extracts and stores each episode's return ('r') from
    the `info['episode']` dict provided by Monitor.
    """
    def __init__(self, verbose=0):
        super(EpisodeInfoCallback, self).__init__(verbose)
        self.episode_lengths = []  # stores episode returns
        self.last_values = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                # record the episode return
                self.episode_lengths.append(info["episode"]["r"])
        return True

    def get_mean_episode_length(self):
        """
        Computes a smoothed mean of recent episode returns.
        """
        recent = self.episode_lengths[-2000:] if self.episode_lengths else []
        if not self.last_values:
            self.last_values = self.episode_lengths[-100:]
        combined = recent + self.last_values
        self.mean_episode_length = np.mean(combined) if combined else 0.0
        self.last_values = self.episode_lengths[-100:]
        return self.mean_episode_length

    def reset_episodes(self):
        """
        Resets the stored episode returns, so only future episodes are considered.
        """
        self.episode_lengths = []
        self.last_values = 0


def new_train_agent_in_world_model_with_eval(
    initial_state_predictor,
    reward_predictor,
    end_predictor,
    wm_predictor,
    total_timesteps=10000,
    eval_interval=2000,
    num_eval_episodes=50
):
    """
    Trains an SAC agent in two world-model settings and evaluates every eval_interval steps:
      1. Three-network world model (separate state, reward, termination nets).
      2. Single-network world model (one_net_predictor).

    At each evaluation point, runs num_eval_episodes on the real UR5 environment
    with the current policy and records the average return.

    Returns the common timestep record and the two average-return lists:
      - timesteps_record
      - three_net_real_returns
      - one_net_real_returns
    """

    # ----------------------------
    # 1) Train & evaluate agent with three separate predictors
    # ----------------------------
    three_net_env = Monitor(
        WorldModelReacherEnv(initial_state_predictor, reward_predictor, end_predictor)
    )
    three_net_model = SAC("MlpPolicy", three_net_env, verbose=0)
    callback = EpisodeInfoCallback()

    timesteps_record = []
    three_net_real_returns = []
    elapsed = 0

    while elapsed < total_timesteps:
        # Train for eval_interval steps in the 3-net world model
        three_net_model.learn(
            total_timesteps=eval_interval,
            reset_num_timesteps=False,
            callback=callback
        )
        elapsed += eval_interval
        timesteps_record.append(elapsed)

        # Evaluate on real UR5
        eval_env = Monitor(NewUR5ReachEnv())
        returns = []
        for _ in range(num_eval_episodes):
            ep_ret = 0
            state, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = three_net_model.predict(state)
                state, reward, done, truncated, info = eval_env.step(action)
                ep_ret += reward
            returns.append(ep_ret)
        eval_env.close()

        avg_return = np.mean(returns)
        three_net_real_returns.append(avg_return)
        print(f"[Three-Net WM] Steps={elapsed} → Real-Env AvgReturn={avg_return:.2f}")

    # ----------------------------
    # 2) Train & evaluate agent with single unified predictor
    # ----------------------------
    one_net_env = Monitor(NewWorldModelReacherEnv(wm_predictor))
    one_net_model = SAC("MlpPolicy", one_net_env, verbose=0)
    callback = EpisodeInfoCallback()

    one_net_real_returns = []
    elapsed = 0

    while elapsed < total_timesteps:
        # Train for eval_interval steps in the 1-net world model
        one_net_model.learn(
            total_timesteps=eval_interval,
            reset_num_timesteps=False,
            callback=callback
        )
        elapsed += eval_interval

        # Evaluate on real UR5
        eval_env = Monitor(NewUR5ReachEnv())
        returns = []
        for _ in range(num_eval_episodes):
            ep_ret = 0
            state, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = one_net_model.predict(state)
                state, reward, done, truncated, info = eval_env.step(action)
                ep_ret += reward
            returns.append(ep_ret)
        eval_env.close()

        avg_return = np.mean(returns)
        one_net_real_returns.append(avg_return)
        print(f"[One-Net WM] Steps={elapsed} → Real-Env AvgReturn={avg_return:.2f}")

    return timesteps_record, three_net_real_returns, one_net_real_returns


# =====================================
# Main entry point
# =====================================
def main():
    all_three_net_returns = []
    all_one_net_returns = []
    rollouts = 50

    for run in range(rollouts):
        print(f"=== Starting rollout {run+1}/{rollouts} ===")
        # Step 1: Collect data from the real UR5
        print("Collecting data from UR5...")
        real_env = NewUR5ReachEnv()
        ppo_agent = SAC("MlpPolicy", env=real_env, verbose=0)
        states, actions, next_states, rewards, dones = collect_data(
            ppo_agent, n_episodes=2000, max_steps=100
        )

        # Step 2: Build DataLoader and train the separate predictors
        dl = prepare_dataloader(states, actions, next_states, rewards, dones, batch_size=64)
        state_predictor   = StatePredictor(state_dim=12, action_dim=6)
        reward_predictor  = RewardPredictor(state_dim=12, action_dim=6)
        term_predictor    = TerminationPredictor(state_dim=12, action_dim=6)
        one_net_predictor = OneNetworkWM(state_dim=12, action_dim=6)

        print("\nTraining state predictor...")
        train_predictor(state_predictor, dl, epochs=5, lr=1e-3)

        print("\nTraining reward predictor...")
        train_reward_predictor(reward_predictor, dl, epochs=5, lr=1e-3)

        print("\nTraining termination predictor...")
        train_termination_predictor(term_predictor, dl, epochs=5, lr=1e-3)

        print("\nTraining unified world model predictor...")
        train_one_net_wm_model(one_net_predictor, dl, epochs=20, lr=1e-3)

        # Step 3: Train SAC in both world-models and evaluate on real UR5
        print("\nTraining and evaluating agents in the world models...")
        timesteps, three_net_returns, one_net_returns = new_train_agent_in_world_model_with_eval(
            state_predictor,
            reward_predictor,
            term_predictor,
            one_net_predictor,
            total_timesteps=40000,
            eval_interval=2000,
            num_eval_episodes=50
        )

        all_three_net_returns.append(three_net_returns)
        all_one_net_returns.append(one_net_returns)

        # Convert to numpy arrays for statistics
        arr_three = np.array(all_three_net_returns)
        arr_one   = np.array(all_one_net_returns)

        if run > 0:
            def mean_ci(data, z=1.96):
                mu = np.mean(data, axis=0)
                sd = np.std(data, axis=0)
                se = sd / np.sqrt(data.shape[0])
                return mu, z * se

            mu3, ci3 = mean_ci(arr_three)
            mu1, ci1 = mean_ci(arr_one)

            plt.figure(figsize=(10, 6))
            plt.plot(timesteps, mu3, color="cyan",
                     label="3-Net WM Agent → Real-Env (50 eps)")
            plt.fill_between(timesteps, mu3 - ci3, mu3 + ci3,
                             color="cyan", alpha=0.2)
            plt.plot(timesteps, mu1, color="orange",
                     label="1-Net WM Agent → Real-Env (50 eps)")
            plt.fill_between(timesteps, mu1 - ci1, mu1 + ci1,
                             color="orange", alpha=0.2)
            plt.xlabel("Training Timesteps")
            plt.ylabel("Mean Episode Return")
            plt.title(f"UR5 Return Comparison (Rollout {run+1}/{rollouts})")
            plt.legend()
            plt.grid(True)
            plt.savefig("network_count_comparison.png")
            plt.show()
            plt.close()



if __name__ == "__main__":
    main()

