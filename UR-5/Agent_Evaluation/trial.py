import gymnasium as gym
from gymnasium.core import ObsType, ActType, SupportsFloat, RenderFrame
from gymnasium.spaces import Space
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import mujoco
from mujoco import viewer
import mujoco_env
#from mujoco_env import MujocoEnv
import math
#from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Tuple, Union, List
import warnings

warnings.filterwarnings(
    "ignore",
    message="WARN: Overwriting existing videos*",
    category=UserWarning
)

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
# 1. Data Collection in UR5
# =====================================
def collect_data(agent, n_episodes=100, max_steps=500):
    """
    Iterates through the UR5 environment using a random policy and stores
    transitions: (state, action, next state).
    """
    env = NewUR5ReachEnv()
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

        # print(f'Episode: {ep}.  Length (steps): {steps}.    Reward: {total_reward}')

    env.close()
    return (
        np.array(states),
        np.array(actions),
        np.array(next_states),
        np.array(rewards),
        np.array(dones)
    )


# =====================================
# 2. State Prediction Network with PyTorch
# =====================================
class StatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        The network takes as input the current state (dim=state_dim) and
        the action in one-hot format (dim=action_dim), and predicts the
        next state (dim=state_dim).
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

        self.fc_angles = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            # nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, state_dim - 6)
        )

    def forward(self, state, action_batch):
        x = torch.cat([state[:, 6:] / (2 * math.pi),
                       action_batch / (2 * math.pi)], dim=1)

        next_angles = (self.fc_angles(x) * (2 * math.pi)) \
            .clamp(min=-2 * math.pi, max=2 * math.pi)

        # Concatenates the representations
        next_joint_angles = next_angles / (2 * math.pi)
        x = torch.cat([state[:, 3:6], next_joint_angles], dim=1)

        # A residual connection is used: current state + predicted delta
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
    the real next state and the predicted one.
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


def evaluate_predictor(model, end_model, agent, n_episodes=10, max_steps=500):
    """
    Evaluates the predictor on real UR5 episodes, computing the MSE
    of the prediction at each step. Reports average error per episode
    and overall.
    """
    env = NewUR5ReachEnv()
    total_errors = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_errors = []
        steps = 0

        while not done and steps < max_steps:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            initial = state_tensor[:, :3]

            # Use a random action for evaluation
            action, _ = agent.predict(state)
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

            model.eval()
            end_model.eval()
            with torch.no_grad():
                initial_pred = model(state_tensor, action_tensor)
                end_pred = end_model(state_tensor, action_tensor)
                pred_next = torch.cat((initial, initial_pred, end_pred), dim=1).squeeze(0).numpy()

            true_next, reward, done, truncated, info = env.step(action)
            error = np.mean((pred_next - true_next) ** 2)
            episode_errors.append(error)

            state = true_next
            steps += 1

        avg_error = np.mean(episode_errors)
        total_errors.append(avg_error)

    env.close()
    print(f"Global average error over {n_episodes} episodes: {np.mean(total_errors):.4f}")


# ====================================
# 3. Reward Prediction Network
# ====================================
class RewardPredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        The network takes as input the current state (dim=state_dim),
        the one-hot action (dim=action_dim), and the next state
        (dim=state_dim), and predicts the termination probability
        (scalar between 0 and 1).
        """
        super(RewardPredictor, self).__init__()

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

        self.fc_special = nn.Sequential(
            nn.Linear((state_dim - 6) + 1, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, next_state, termination):
        logits = self.fc(-next_state[:, :6])
        special = self.fc_special(torch.cat([next_state[:, :6], termination], dim=1))
        reward = special - logits
        return reward.squeeze(1)

def train_reward_predictor(model, dataloader, epochs=20, lr=1e-3, verbose=1):
    """
    Trains the reward predictor network using SmoothL1Loss to minimize
    the difference between the predicted reward and the true label (0 or 1).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for batch in dataloader:
            state_batch, action_batch, next_state_batch, reward_batch, termination_batch = batch
            optimizer.zero_grad()
            pred_reward = model(next_state_batch, termination_batch)
            loss = criterion(pred_reward, reward_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * state_batch.size(0)
        
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
        and predicts the probability of termination (scalar between 0 and 1).
        """
        super(TerminationPredictor, self).__init__()
        self.action_dim = action_dim  # stored for one-hot encoding if needed
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
        logits = self.fc(next_state[:, :6])
        # apply sigmoid to get a probability
        return torch.sigmoid(logits)


def train_termination_predictor(model, dataloader, epochs=20, lr=1e-3, verbose=1):
    """
    Trains the termination predictor network using BCELoss to minimize
    the difference between the predicted probability and the true label (0 or 1).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for batch in dataloader:
            state_batch, action_batch, next_state_batch, reward_batch, termination_batch = batch
            optimizer.zero_grad()
            pred_term = model(next_state_batch)
            loss = criterion(pred_term, termination_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * state_batch.size(0)
        
        epoch_loss /= len(dataloader.dataset)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# =====================================
# 5. Define the world-model environment using the predictor networks
# =====================================
class WorldModelReacherEnv(gym.Env):
    """
    A simulated environment (world model) that uses the predictor networks
    to evolve the state. Reward and termination logic follow the same rules
    as in the CartPole-v1 environment.
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 state_predictor,
                 reward_predictor,
                 end_predictor,
                 state_dim=12,
                 action_dim=6):
        super(WorldModelReacherEnv, self).__init__()
        self.state_predictor = state_predictor
        self.reward_predictor = reward_predictor
        self.end_predictor = end_predictor
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
            low=-action_max, high=+action_max, shape=(action_dim,)
        )
        obs_low = np.array([-1.0] * 3 + [-1.0] * 3 + [-2*math.pi]*6, dtype=np.float32)
        obs_high = np.array([+1.0] * 3 + [+1.0] * 3 + [+2*math.pi]*6, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, shape=(state_dim,)
        )

        self.total_reward = 0.0
        self.steps = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        """
        Resets the simulation. Samples a new target position, resets counters,
        and returns the initial observation.
        """
        self.steps = 0
        self.total_reward = 0.0

        # Sample a random target in x,y ∈ [-0.5, 0.5], z = 0.7
        self.target_pos = np.random.uniform(-0.5, 0.5, size=3)
        self.target_pos[2] = 0.7

        self.state = self.get_observation()
        return self.state, {}

    def get_observation(self) -> np.ndarray:
        """
        Reads the current end-effector position and the joint angles,
        then concatenates: [target_pos (3), ee_pos (3), joint_angles (6)].
        """
        mujoco.mj_step(self.model, self.data)  # ensure data.xpos, data.qpos are up to date
        ee_pos = self.data.xpos[self.ee_link_id]
        joint_pos = self.data.qpos[:6]
        return np.concatenate((self.target_pos, ee_pos, joint_pos), dtype=np.float32)

    def wait_until_stable(self, sim_steps=500) -> bool:
        """
        Advances the MuJoCo simulation until joint velocities have settled,
        or until `sim_steps` steps have elapsed.
        """
        def _read_obs():
            return np.concatenate((self.data.xpos[self.ee_link_id],
                                   self.data.qpos[:6]), axis=0)

        prev_obs = _read_obs()
        for _ in range(sim_steps):
            mujoco.mj_step(self.model, self.data)
            new_obs = _read_obs()
            if np.sum(np.abs(prev_obs - new_obs)) < 5e-3:
                return True
            prev_obs = new_obs
        return False

    def step(self, action):
        """
        Applies `action` in the simulated model, then uses the predictor networks
        to estimate the next state, reward, and termination flag.
        Every 5 steps, we inject the true observation to reduce model drift.
        """
        self.steps += 1

        # Apply and clip the control signals
        self.data.ctrl[:] += action
        self.data.ctrl[:] = np.clip(self.data.ctrl, -2*math.pi, 2*math.pi)

        self.wait_until_stable()

        # Prepare tensors for the predictors
        state_tensor = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

        self.state_predictor.eval()
        self.reward_predictor.eval()
        self.end_predictor.eval()
        with torch.no_grad():
            # Predict the next state
            next_state_tensor = self.state_predictor(state_tensor, action_tensor)
            next_state = next_state_tensor.squeeze(0).numpy()

            # Every 5 steps, replace with a real simulator observation
            if self.steps % 5 == 0:
                next_state = self.get_observation()
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Determine termination via the end_predictor network
            term_prob = self.end_predictor(next_state_tensor)
            terminated = bool((term_prob >= 0.5).item())
            term_tensor = term_prob

            # Predict the reward
            reward_tensor = self.reward_predictor(next_state_tensor, term_tensor)
            reward = reward_tensor.squeeze(0).item()

        self.state = next_state
        self.total_reward += reward
        truncated = (self.steps >= self.max_steps)

        info = {}
        if terminated or truncated:
            info["episode"] = {"l": self.steps, "r": self.total_reward}

        return self.state, reward, terminated, truncated, info


# =====================================
# 6. Training the PPO agent in the world model
# =====================================
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeInfoCallback(BaseCallback):
    """
    A callback to extract and store each episode's return. The Monitor wrapper
    includes an 'episode' entry in info with keys 'l' (length) and 'r' (return).
    """
    def __init__(self, verbose=0):
        super(EpisodeInfoCallback, self).__init__(verbose)
        self.episode_returns = []
        self.last_buffer = []
        self.mean_return = 0.0

    def _on_step(self) -> bool:
        """
        Called at every environment step. We scan the 'infos' list for completed
        episodes and record their returns.
        """
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_returns.append(info["episode"]["r"])
        return True

    def get_mean_episode_return(self):
        """
        Computes a smoothed mean over the last 2000 returns, blending with the
        previous buffer to stabilize the metric.
        """
        recent = self.episode_returns[-2000:]
        if not self.last_buffer:
            self.last_buffer = self.episode_returns[-100:]
        combined = recent + self.last_buffer
        self.mean_return = float(np.mean(combined))
        self.last_buffer = self.episode_returns[-100:]
        return self.mean_return

    def reset_episodes(self):
        """Clears all recorded episode returns."""
        self.episode_returns = []
        self.last_buffer = []
        self.mean_return = 0.0


import imageio

def new_train_agent_in_world_model_with_eval(
    state_predictor,
    reward_predictor,
    end_predictor,
    total_timesteps=10000,
    eval_interval=2000,
    num_eval_episodes=50,
    update_interval=10000
):
    """
    Trains an agent with SAC in the world model (simulated environment) and, every
    eval_interval steps, evaluates the current policy in three scenarios:
      1. During training in the world model (using the callback’s average return).
      2. By simulating num_eval_episodes in the world model with the trained agent.
      3. By simulating num_eval_episodes in the real UR5 environment with that agent.

    In parallel, it also trains a new agent directly in the real environment and
    logs its average return every eval_interval steps.

    Metrics from all four curves are collected and plotted in
    'episode_length_comparison.png'.
    """

    # ----------------------------
    # 1. Train the agent in the world model
    # ----------------------------
    training_env = Monitor(
        WorldModelReacherEnv(state_predictor, reward_predictor, end_predictor)
    )
    model = SAC("MlpPolicy", training_env, verbose=0)
    episode_callback = EpisodeInfoCallback()

    wm_timesteps_record = []    # Common X-axis for the world-model agent
    wm_training_rewards = []    # Callback metric during world-model training
    wm_sim_world_rewards = []   # Eval: 50 episodes in the world model
    wm_sim_real_rewards = []    # Eval: 50 episodes in the real environment

    total_evaluated_timesteps = 0
    os.environ["MUJOCO_GL"] = "egl"

    while total_evaluated_timesteps < total_timesteps:
        # Re-wrap environment so Monitor resets internally
        training_env = Monitor(
            WorldModelReacherEnv(state_predictor, reward_predictor, end_predictor)
        )
        model.set_env(training_env)

        # Train for eval_interval steps in the world model
        model.learn(
            total_timesteps=eval_interval,
            reset_num_timesteps=False,
            callback=episode_callback
        )
        current_training_reward = episode_callback.get_mean_episode_length()
        total_evaluated_timesteps += eval_interval

        wm_timesteps_record.append(total_evaluated_timesteps)
        wm_training_rewards.append(current_training_reward)

        # --- Evaluation in the world model (num_eval_episodes) ---
        sim_wm_rewards = []
        wm_eval_env = WorldModelReacherEnv(
            state_predictor, reward_predictor, end_predictor
        )
        for _ in range(num_eval_episodes):
            ep_reward = 0
            state, _ = wm_eval_env.reset()
            while True:
                action, _ = model.predict(state)
                state, reward, done, truncated, info = wm_eval_env.step(action)
                ep_reward += reward
                if done or truncated:
                    break
            sim_wm_rewards.append(ep_reward)
        wm_sim_world_rewards.append(np.mean(sim_wm_rewards))
        wm_eval_env.close()

        # --- Evaluation in the real UR5 environment (num_eval_episodes) ---
        sim_real_rewards = []
        real_eval_env = Monitor(NewUR5ReachEnv(render_mode='rgb_array'))
        for _ in range(num_eval_episodes):
            ep_reward = 0
            state, _ = real_eval_env.reset()
            while True:
                action, _ = model.predict(state)
                state, reward, done, truncated, info = real_eval_env.step(action)
                ep_reward += reward
                if done or truncated:
                    break
            sim_real_rewards.append(ep_reward)
        wm_sim_real_rewards.append(np.mean(sim_real_rewards))
        real_eval_env.close()

        print(
            f"[World Model Agent] Steps: {total_evaluated_timesteps} | "
            f"Train Return: {current_training_reward:.2f} | "
            f"WorldModel Eval (50): {wm_sim_world_rewards[-1]:.2f} | "
            f"Real Env Eval (50): {wm_sim_real_rewards[-1]:.2f}"
        )

        # Periodically retrain the world model with fresh data
        if (total_evaluated_timesteps % update_interval == 0
                and total_evaluated_timesteps != total_timesteps):
            print(f"Retraining world model at step {total_evaluated_timesteps}")
            estados, acciones, siguientes_estados, recompensas, terminaciones = collect_data(
                model, n_episodes=1000, max_steps=100
            )

            # Step 2: Prepare DataLoader and train new predictor networks
            new_state_predictor = StatePredictor(state_dim=12, action_dim=6)
            new_reward_predictor = RewardPredictor(state_dim=12, action_dim=6)
            new_end_predictor = TerminationPredictor(state_dim=12, action_dim=6)

            dataloader = prepare_dataloader(
                estados, acciones, siguientes_estados,
                recompensas, terminaciones,
                batch_size=128
            )
            train_predictor(new_state_predictor, dataloader, epochs=10, lr=1e-3, verbose=0)
            state_predictor = evaluate_state_predictor(
                state_predictor, new_state_predictor, dataloader
            )

            train_reward_predictor(new_reward_predictor, dataloader, epochs=5, lr=1e-3, verbose=0)
            reward_predictor = evaluate_reward_predictor(
                reward_predictor, new_reward_predictor, dataloader
            )

            train_termination_predictor(new_end_predictor, dataloader, epochs=5, lr=1e-3, verbose=0)
            end_predictor = evaluate_termination_predictor(
                end_predictor, new_end_predictor, dataloader
            )

    training_env.close()

    # ----------------------------
    # 2. Train a new agent in the real environment
    # ----------------------------
    real_training_env = Monitor(NewUR5ReachEnv())
    real_agent = SAC("MlpPolicy", real_training_env, verbose=0)
    episode_callback_real = EpisodeInfoCallback()

    real_timesteps_record = []  # X-axis for the real-env agent
    real_training_rewards = []  # Mean return during real-env training
    real_eval_rewards = []      # Mean return evaluating 50 episodes in real env

    total_evaluated_real_timesteps = 0
    os.environ["MUJOCO_GL"] = "egl"

    while total_evaluated_real_timesteps < total_timesteps:
        real_agent.learn(
            total_timesteps=eval_interval,
            reset_num_timesteps=False,
            callback=episode_callback_real
        )
        current_real_reward = episode_callback_real.get_mean_episode_length()
        total_evaluated_real_timesteps += eval_interval

        real_timesteps_record.append(total_evaluated_real_timesteps)
        real_training_rewards.append(current_real_reward)

        # Evaluate the real-agent in the real environment
        sim_real_agent_rewards = []
        real_eval_env = Monitor(NewUR5ReachEnv(render_mode='rgb_array'))
        for _ in range(num_eval_episodes):
            ep_reward = 0
            state, _ = real_eval_env.reset()
            while True:
                action, _ = real_agent.predict(state)
                state, reward, done, truncated, info = real_eval_env.step(action)
                ep_reward += reward
                if done or truncated:
                    break
            sim_real_agent_rewards.append(ep_reward)

        mean_eval_reward = np.mean(sim_real_agent_rewards)
        real_eval_rewards.append(mean_eval_reward)
        real_eval_env.close()

        print(
            f"[Real Agent] Steps: {total_evaluated_real_timesteps} | "
            f"Train Return: {current_real_reward:.2f} | "
            f"Eval Return (50): {mean_eval_reward:.2f}"
        )

    real_training_env.close()

    return wm_timesteps_record, wm_training_rewards, wm_sim_world_rewards, wm_sim_real_rewards, real_training_rewards, real_eval_rewards

def evaluate_state_predictor(old_pos_model, new_pos_model, dataloader):
    old_pos_model.eval()
    new_pos_model.eval()

    criterion = nn.MSELoss()

    old_mse = 0
    new_mse = 0

    with torch.no_grad():
        for state_batch, action_batch, next_state_batch, reward_batch, termination_batch in dataloader:

            old_preds = old_pos_model(state_batch, action_batch)

            old_mse += criterion(old_preds, next_state_batch).item()

            new_preds = new_pos_model(state_batch, action_batch)

            new_mse += criterion(new_preds, next_state_batch).item()

    if old_mse < new_mse:
        selected_pos_model = old_pos_model
        print(f"\tOld state predictor selected with MSE: {old_mse:.4f} (New with {new_mse:.4f})")
    if old_mse >= new_mse:
        selected_pos_model = new_pos_model
        print(f"\tNew state predictor selected with MSE: {new_mse:.4f} (Old with {old_mse:.4f})")
    return selected_pos_model

def evaluate_reward_predictor(old_model, new_model, dataloader):
    old_model.eval()
    new_model.eval()

    criterion = nn.MSELoss()

    old_mse = 0
    new_mse = 0

    with torch.no_grad():
        for state_batch, action_batch, next_state_batch, reward_batch, termination_batch in dataloader:

            old_preds = old_model(next_state_batch).squeeze(-1)

            old_mse += criterion(old_preds, reward_batch.squeeze(-1)).item()

            new_preds = new_model(next_state_batch).squeeze(-1)

            new_mse += criterion(new_preds, reward_batch.squeeze(-1)).item()

    if old_mse < new_mse:
        selected_model = old_model
        print(f"\tOld reward predictor selected with MSE: {old_mse:.4f} (New with {new_mse:.4f})")
    if old_mse >= new_mse:
        selected_model = new_model
        print(f"\tNew reward predictor selected with MSE: {new_mse:.4f} (Old with {old_mse:.4f})")
    return selected_model

def evaluate_termination_predictor(old_model, new_model, dataloader):
    old_model.eval()
    new_model.eval()
    old_correct = 0
    old_total = 0
    new_correct = 0
    new_total = 0

    with torch.no_grad():
        for state_batch, action_batch, next_state_batch, reward_batch, termination_batch in dataloader:

            old_preds = old_model(next_state_batch)
            old_preds = old_preds.squeeze(-1)
            old_preds = (old_preds >= 0.5).long()

            old_correct += (old_preds == termination_batch.squeeze(-1)).sum().item()
            old_total   += termination_batch.size(0)

            new_preds = new_model(next_state_batch)
            new_preds = new_preds.squeeze(-1)
            new_preds = (new_preds >= 0.5).long()

            new_correct += (new_preds == termination_batch.squeeze(-1)).sum().item()
            new_total   += termination_batch.size(0)

    old_accuracy = old_correct / old_total
    new_accuracy = new_correct / new_total
    if old_accuracy >= new_accuracy:
        selected_model = old_model
        print(f"\tOld termination predictor selected with accuracy: {old_accuracy*100:.2f}% (New with {new_accuracy*100:.2f}%)")
    if old_accuracy < new_accuracy:
        selected_model = new_model
        print(f"\tNew termination predictor selected with accuracy: {new_accuracy*100:.2f}% (Old with {old_accuracy*100:.2f}%)")
    return selected_model

# =====================================
# Main function
# =====================================
def main():
    all_wm_training_returns = []   # World Model: training returns (from callback)
    all_wm_eval_world_returns = [] # World Model eval: 50 episodes in the world model
    all_wm_eval_real_returns = []  # World Model eval: 50 episodes in the real env
    all_real_training_returns = [] # Real env: training returns
    all_real_eval_returns = []     # Real env: eval returns

    rollouts = 50

    for i in range(rollouts):
        print(f"=== Starting training run {i+1}/{rollouts} ===")

        # Step 1: Collect data from the real UR5
        print("Collecting data from UR5...")
        real_env = NewUR5ReachEnv()
        agent = SAC("MlpPolicy", env=real_env, verbose=0)

        states, actions, next_states, rewards, dones = collect_data(
            agent,
            n_episodes=2000,
            max_steps=100
        )

        # Step 2: Prepare DataLoader and train the predictor networks
        dataloader = prepare_dataloader(
            states,
            actions,
            next_states,
            rewards,
            dones,
            batch_size=128
        )

        state_predictor = StatePredictor(state_dim=12, action_dim=6)
        reward_predictor = RewardPredictor(state_dim=12, action_dim=6)
        termination_predictor = TerminationPredictor(state_dim=12, action_dim=6)

        print("\nTraining the state prediction network...")
        train_predictor(state_predictor, dataloader, epochs=5, lr=1e-3)

        print("\nEvaluating the state predictions...")
        # evaluate_predictor(state_predictor, end_state_predictor, ppo, n_episodes=100, max_steps=100)

        print("\nTraining the reward prediction network...")
        train_reward_predictor(reward_predictor, dataloader, epochs=5, lr=1e-3)

        print("\nTraining the termination prediction network...")
        train_termination_predictor(termination_predictor, dataloader, epochs=5, lr=1e-3)

        # Step 3: Train a SAC agent in the World Model and evaluate
        print("\nTraining the SAC agent in the World Model...")
        (
            timesteps_record,
            wm_training_returns,
            wm_eval_world_returns,
            wm_eval_real_returns,
            real_training_returns,
            real_eval_returns
        ) = new_train_agent_in_world_model_with_eval(
            state_predictor,
            reward_predictor,
            termination_predictor,
            total_timesteps=40000,
            eval_interval=2000,
            num_eval_episodes=50,
            update_interval=40000
        )

        # Aggregate results from this rollout
        all_wm_training_returns.append(wm_training_returns)
        all_wm_eval_world_returns.append(wm_eval_world_returns)
        all_wm_eval_real_returns.append(wm_eval_real_returns)
        all_real_training_returns.append(real_training_returns)
        all_real_eval_returns.append(real_eval_returns)

        # Convert to numpy arrays for statistics
        plot_wm_training = np.array(all_wm_training_returns)
        plot_wm_world_eval = np.array(all_wm_eval_world_returns)
        plot_wm_real_eval = np.array(all_wm_eval_real_returns)
        plot_real_training = np.array(all_real_training_returns)
        plot_real_eval = np.array(all_real_eval_returns)

        if i > 0:
            # Compute means and 95% confidence intervals
            def compute_mean_ci(data, z=1.96):
                means = np.mean(data, axis=0)
                stds = np.std(data, axis=0)
                sem = stds / np.sqrt(data.shape[0])
                ci = z * sem
                return means, ci

            wm_train_mean, wm_train_ci     = compute_mean_ci(plot_wm_training)
            wm_wm_eval_mean, wm_wm_eval_ci = compute_mean_ci(plot_wm_world_eval)
            wm_real_eval_mean, wm_real_eval_ci = compute_mean_ci(plot_wm_real_eval)
            real_train_mean, real_train_ci = compute_mean_ci(plot_real_training)
            real_eval_mean, real_eval_ci   = compute_mean_ci(plot_real_eval)

            # Plot evaluation returns
            plt.figure(figsize=(10, 6))
            plt.plot(
                timesteps_record,
                wm_wm_eval_mean,
                color="orange",
                label="WM Agent – Eval in World Model (50 eps)"
            )
            plt.fill_between(
                timesteps_record,
                wm_wm_eval_mean - wm_wm_eval_ci,
                wm_wm_eval_mean + wm_wm_eval_ci,
                color="orange",
                alpha=0.2
            )
            plt.plot(
                timesteps_record,
                wm_real_eval_mean,
                color="green",
                label="WM Agent – Eval in Real Env (50 eps)"
            )
            plt.fill_between(
                timesteps_record,
                wm_real_eval_mean - wm_real_eval_ci,
                wm_real_eval_mean + wm_real_eval_ci,
                color="green",
                alpha=0.2
            )
            plt.plot(
                timesteps_record,
                real_eval_mean,
                color="purple",
                label="Real Agent Evaluation (50 eps)"
            )
            plt.fill_between(
                timesteps_record,
                real_eval_mean - real_eval_ci,
                real_eval_mean + real_eval_ci,
                color="purple",
                alpha=0.2
            )
            plt.xlabel("Training Timesteps")
            plt.ylabel("Mean Episode Return")
            plt.title(f"UR5 Evaluation Comparison (Run {i+1})")
            plt.legend()
            plt.grid(True)
            plt.savefig("new_evaluation_reward_comparison.png")
            plt.show()
            plt.close()

            # Plot training returns
            plt.figure(figsize=(10, 6))
            plt.plot(
                timesteps_record,
                wm_train_mean,
                color="blue",
                label="World Model Training Return"
            )
            plt.fill_between(
                timesteps_record,
                wm_train_mean - wm_train_ci,
                wm_train_mean + wm_train_ci,
                color="blue",
                alpha=0.2
            )
            plt.plot(
                timesteps_record,
                real_train_mean,
                color="red",
                label="Real Agent Training Return"
            )
            plt.fill_between(
                timesteps_record,
                real_train_mean - real_train_ci,
                real_train_mean + real_train_ci,
                color="red",
                alpha=0.2
            )
            plt.xlabel("Training Timesteps")
            plt.ylabel("Mean Episode Return")
            plt.title(f"UR5 Training Comparison (Run {i+1})")
            plt.legend()
            plt.grid(True)
            plt.savefig("new_training_reward_comparison.png")
            plt.show()
            plt.close()

if __name__ == "__main__":
    main()

