import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt

# =====================================
# 1. Data Collection in Reacher-v5
# =====================================
def collect_data(ppo, n_episodes=100, max_steps=500):
    """
    Runs the Reacher-v5 environment using a random policy and stores
    transitions as tuples: (state, action, next_state, reward).
    """
    env = gym.make("Reacher-v5")
    states = []
    actions = []
    next_states = []
    rewards = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        for t in range(max_steps):
            if done:
                break
            action, _ = ppo.predict(state)
            next_state, reward, done, truncated, info = env.step(action)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            state = next_state
    env.close()
    return (np.array(states),
            np.array(actions),
            np.array(next_states),
            np.array(rewards))


# =====================================
# 2. State-Prediction Network with PyTorch
# =====================================
class StatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Takes current state (dim=state_dim) and a one-hot action vector
        (dim=action_dim), and predicts the next state (dim=state_dim).
        """
        super(StatePredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim // 2, state_dim)
        )
    
    def forward(self, state, action):
        # concatenate state and one-hot action
        x = torch.cat([state.float(), action], dim=1)
        # predict the change and add to current state
        return state + self.fc(x)


def prepare_dataloader(states, actions, next_states, batch_size=32):
    """
    Prepares a DataLoader from the collected transitions.
    """
    states_t = torch.tensor(states, dtype=torch.float64)
    actions_t = torch.tensor(actions, dtype=torch.float32)
    next_states_t = torch.tensor(next_states, dtype=torch.float64)
    dataset = TensorDataset(states_t, actions_t, next_states_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_predictor(model, dataloader, epochs=20, lr=1e-3):
    """
    Trains the state predictor network using MSELoss to minimize
    the difference between the true next state and the predicted one.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for state_b, action_b, next_state_b in dataloader:
            optimizer.zero_grad()
            pred_next = model(state_b, action_b)
            loss = criterion(pred_next, next_state_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * state_b.size(0)
        epoch_loss /= len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")


def evaluate_predictor(model, ppo, n_episodes=10, max_steps=50):
    """
    Evaluates the state predictor on real Reacher-v5 episodes,
    computing MSE at each step. Reports average error per episode and overall.
    """
    env = gym.make("Reacher-v5")
    total_errors = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_errors = []
        steps = 0
        while not done and steps < max_steps:
            state_t = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
            # take a policy action for evaluation
            action, _ = ppo.predict(state)
            action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            model.eval()
            with torch.no_grad():
                pred_next = model(state_t, action_t).squeeze(0).numpy()
            true_next, reward, done, truncated, info = env.step(action)
            error = np.mean((pred_next - true_next) ** 2)
            episode_errors.append(error)
            state = true_next
            steps += 1
        mean_err = np.mean(episode_errors)
        total_errors.append(mean_err)
    env.close()
    print("Overall average MSE over", n_episodes, "episodes:", np.mean(total_errors))

# ====================================
# 3. Termination (Reward) Prediction Network
# ====================================
class RewardPredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Input:
          - current state (dim = state_dim)
          - action vector (one-hot, dim = action_dim)
          - next state (dim = state_dim)
        Output:
          - a scalar logit for the termination probability (before sigmoid)
        """
        super(RewardPredictor, self).__init__()
        self.action_dim = action_dim
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state, action, next_state):
        # concatenate state, action, and next_state
        x = torch.cat([state.float(), action, next_state.float()], dim=1)
        logits = self.fc(x)
        return logits.squeeze(1)


def prepare_dataloader_reward(states, actions, next_states, rewards, batch_size=32):
    """
    Create a DataLoader from collected data.

    Parameters:
      - states:       array of shape [N, state_dim]
      - actions:      array of shape [N, action_dim] (one-hot or raw)
      - next_states:  array of shape [N, state_dim]
      - rewards:      array of shape [N] (0/1 termination labels or scalar rewards)
      - batch_size:   batch size for training

    Returns:
      - a PyTorch DataLoader yielding (state, action, next_state, reward)
    """
    st = torch.tensor(states, dtype=torch.float64)
    ac = torch.tensor(actions, dtype=torch.float32)
    ns = torch.tensor(next_states, dtype=torch.float64)
    rw = torch.tensor(rewards, dtype=torch.float32)
    dataset = TensorDataset(st, ac, ns, rw)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_reward_predictor(model, dataloader, epochs=20, lr=1e-3):
    """
    Train the termination-prediction network.

    Uses MSELoss (or BCELoss if you change it) between the predicted logit
    and the actual label (0 or 1).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for state_b, action_b, next_state_b, label_b in dataloader:
            optimizer.zero_grad()
            pred_logit = model(state_b, action_b, next_state_b)
            loss = criterion(pred_logit, label_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * state_b.size(0)
        epoch_loss /= len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")


# =====================================
#   Single Network World Model (OneNet)
# =====================================
class OneNetworkWM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        A single network that, given current state and action (one-hot),
        predicts:
          - the next state (state + estimated delta)
          - the scalar reward
        """
        super(OneNetworkWM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # shared trunk processing state + action
        self.shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.05)
        )
        # head for state delta
        self.state_head = nn.Linear(hidden_dim // 2, state_dim)
        # head for reward
        self.reward_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, state, action):
        # concatenate state and action
        x = torch.cat([state.float(), action], dim=1)
        features = self.shared(x)

        # predict next state = state + delta
        delta = self.state_head(features)
        next_state = state + delta

        # predict reward
        reward = self.reward_head(features).squeeze(1)
        return next_state, reward


def train_one_net_wm_model(model, dataloader, epochs=20, lr=1e-3, device='cpu'):
    """
    Train the unified world-model network using:
      - MSELoss for next-state prediction
      - MSELoss for reward prediction

    Assumes each batch supplies: state, action, next_state, reward_label.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        for state_b, action_b, next_state_b, reward_b in dataloader:
            # move to device
            state_b = state_b.to(device)
            action_b = action_b.to(device)
            next_state_b = next_state_b.to(device)
            reward_b = reward_b.to(device)

            optimizer.zero_grad()
            pred_next, pred_reward = model(state_b, action_b)
            loss_state = criterion(pred_next, next_state_b)
            loss_reward = criterion(pred_reward, reward_b)
            loss = loss_state + loss_reward
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * state_b.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# =====================================
# 4. Define the world-model environment using the predictor network
# =====================================
class WorldModelReacherEnv(gym.Env):
    """
    A simulated environment (world model) that uses predictor networks
    to evolve the state. Reward and termination logic mirror CartPole-v1.
    """
    metadata = {"render_modes": []}

    def __init__(self, state_predictor, reward_predictor, state_dim=10, action_dim=2):
        super(WorldModelReacherEnv, self).__init__()
        self.state_predictor = state_predictor
        self.reward_predictor = reward_predictor
        self.state_dim = state_dim
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float64
        )
        self.total_reward = 0.0
        self.steps = 0
        self.max_steps = 50

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.total_reward = 0.0  # reset accumulated reward
        self.state = np.zeros(self.state_dim)  # initial balanced state
        return self.state, {}

    def step(self, action):
        self.steps += 1
        st_t = torch.tensor(self.state, dtype=torch.float64).unsqueeze(0)
        ac_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

        self.state_predictor.eval()
        self.reward_predictor.eval()
        with torch.no_grad():
            next_state = self.state_predictor(st_t, ac_t).squeeze(0).numpy()
            ns_t = torch.tensor(next_state, dtype=torch.float64).unsqueeze(0)
            reward = self.reward_predictor(st_t, ac_t, ns_t).squeeze(0).item()

        self.state = next_state
        self.total_reward += reward
        terminated = False
        truncated = (self.steps >= self.max_steps)
        info = {}
        if terminated or truncated:
            info["episode"] = {"l": self.steps, "r": self.total_reward}
        return self.state, reward, terminated, truncated, info


#
#   New single-network world model
#
class NewWorldModelReacherEnv(gym.Env):
    """
    A simulated environment (world model) driven by a single predictor network.
    Uses the same reward and termination criteria as CartPole-v1.
    """
    metadata = {"render_modes": []}

    def __init__(self, wm_predictor, state_dim=10, action_dim=2):
        super(NewWorldModelReacherEnv, self).__init__()
        self.wm_predictor = wm_predictor
        self.state_dim = state_dim
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float64
        )
        self.total_reward = 0.0
        self.steps = 0
        self.max_steps = 50

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.total_reward = 0.0
        self.state = np.zeros(self.state_dim)
        return self.state, {}

    def step(self, action):
        self.steps += 1
        st_t = torch.tensor(self.state, dtype=torch.float64).unsqueeze(0)
        ac_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

        self.wm_predictor.eval()
        with torch.no_grad():
            next_state_t, reward = self.wm_predictor(st_t, ac_t)
        next_state = next_state_t.squeeze(0).numpy()

        self.state = next_state
        self.total_reward += reward
        terminated = False
        truncated = (self.steps >= self.max_steps)
        info = {}
        if terminated or truncated:
            info["episode"] = {"l": self.steps, "r": self.total_reward}
        return self.state, reward, terminated, truncated, info


# =====================================
# 5. Train a PPO agent in the world model
# =====================================
def train_agent_in_world_model(state_predictor, reward_predictor, timesteps=10000):
    """
    Train a PPO agent with stable-baselines3 on the simulated world-model
    environment built from the predictor networks.
    """
    env = WorldModelReacherEnv(state_predictor, reward_predictor)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    env.close()
    return model


# =====================================
# 6. Test the trained policy in CartPole-v1 and record 5 episodes
# =====================================
def test_policy(model, n_episodes=5, video_folder="./videos"):
    """
    Runs the trained policy in the real CartPole-v1 environment.
    Uses RecordVideo to save each episode to disk.
    """
    os.makedirs(video_folder, exist_ok=True)
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda ep: True)

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(state)
            state, reward, done, terminated, info = env.step(action)
            total_reward += reward
        print(f"Test Episode {ep+1} - Total Reward: {total_reward}")
    env.close()


from stable_baselines3.common.callbacks import BaseCallback

class EpisodeInfoCallback(BaseCallback):
    """
    Callback to extract and store the length (number of steps) of each episode.
    The Monitor wrapper adds, inside the `info` dict under the 'episode' key,
    a dict that includes, among other values, 'l' (the episode length in steps).
    """
    def __init__(self, verbose=0):
        super(EpisodeInfoCallback, self).__init__(verbose)
        self.episode_lengths = []  # List to accumulate episode lengths
        self.last_values = 0

    def _on_step(self) -> bool:
        # Retrieve the "infos" list from the rollout data
        infos = self.locals.get("infos", [])
        for info in infos:
            # When 'episode' appears in info, an episode has ended
            if "episode" in info:
                self.episode_lengths.append(info["episode"]["l"])
        return True

    def get_mean_episode_length(self):
        # Use up to the last 2,000 lengths for the new mean
        recent = self.episode_lengths[-2000:] if self.episode_lengths else []
        
        if self.last_values == 0:
            # Initialize the smoothing buffer with the last 100 episodes
            self.last_values = self.episode_lengths[-100:]
        
        # Compute mean over recent + previous batch
        combined = recent + self.last_values
        self.mean_episode_length = np.mean(combined) if combined else 0.0
        
        # Update the smoothing buffer
        self.last_values = self.episode_lengths[-100:]
        return self.mean_episode_length

    def reset_episodes(self):
        """Reset episode statistics so only new episodes are counted."""
        self.episode_lengths = []


def new_train_agent_in_world_model_with_eval(
    state_predictor,
    end_predictor,
    wm_predictor,
    total_timesteps=10000,
    eval_interval=2000,
    num_eval_episodes=50
):
    """
    Trains a PPO agent in the world model (simulated env) and, every eval_interval steps:
      1. Records the mean episode length during world-model training (via callback).
      2. Simulates 50 episodes in the world model and records mean reward.
      3. Simulates 50 episodes in the real environment and records mean reward.

    Also trains a new PPO agent directly in the real env, recording its mean
    training length (via callback) and evaluating it every eval_interval steps
    over 50 episodes.

    Metrics are collected and plotted in 'episode_length_comparison.png'.
    """

    # ----------------------------
    # Train 3-network world-model agent
    # ----------------------------
    three_net_env = Monitor(WorldModelReacherEnv(state_predictor, end_predictor))
    three_net_model = PPO("MlpPolicy", three_net_env, verbose=0)
    callback3 = EpisodeInfoCallback()

    wm_timesteps_record = []       # X-axis for world-model agent
    wm_sim_real_rewards = []       # Mean reward over 50 real-env episodes

    total_steps = 0
    while total_steps < total_timesteps:
        # Train for eval_interval steps
        three_net_model.learn(
            total_timesteps=eval_interval,
            reset_num_timesteps=False,
            callback=callback3
        )
        total_steps += eval_interval
        wm_timesteps_record.append(total_steps)

        # Evaluate in real environment (50 episodes)
        sim_real = []
        real_env = Monitor(gym.make("Reacher-v5"))
        for _ in range(num_eval_episodes):
            ep_reward = 0
            state, _ = real_env.reset()
            done = False
            while not done:
                action, _ = three_net_model.predict(state)
                state, reward, done, truncated, info = real_env.step(action)
                ep_reward += reward
            sim_real.append(ep_reward)
        wm_sim_real_rewards.append(np.mean(sim_real))
        real_env.close()

        print(f"[3-Net WM Agent] Steps: {total_steps} -> "
              f"Real-Env Eval (50 eps.) Reward: {wm_sim_real_rewards[-1]:.2f}")

    # ----------------------------
    # Train 1-network world-model agent
    # ----------------------------
    one_net_env = Monitor(NewWorldModelReacherEnv(wm_predictor))
    one_net_model = PPO("MlpPolicy", one_net_env, verbose=0)
    callback1 = EpisodeInfoCallback()

    wm1_timesteps_record = []      # X-axis for single-net agent
    wm1_sim_real_rewards = []      # Mean reward over 50 real-env episodes

    total_steps = 0
    while total_steps < total_timesteps:
        one_net_model.learn(
            total_timesteps=eval_interval,
            reset_num_timesteps=False,
            callback=callback1
        )
        total_steps += eval_interval
        wm1_timesteps_record.append(total_steps)

        sim_real = []
        real_env = Monitor(gym.make("Reacher-v5"))
        for _ in range(num_eval_episodes):
            ep_reward = 0
            state, _ = real_env.reset()
            done = False
            while not done:
                action, _ = one_net_model.predict(state)
                state, reward, done, truncated, info = real_env.step(action)
                ep_reward += reward
            sim_real.append(ep_reward)
        wm1_sim_real_rewards.append(np.mean(sim_real))
        real_env.close()

        print(f"[1-Net WM Agent] Steps: {total_steps} -> "
              f"Real-Env Eval (50 eps.) Reward: {wm1_sim_real_rewards[-1]:.2f}")

    return wm_timesteps_record, wm_sim_real_rewards, wm1_sim_real_rewards


# =====================================
# Main function
# =====================================
def main():
    all_wm3_real_eval_rewards = []   # 3-network world-model agent: real-env eval rewards
    all_wm1_real_eval_rewards = []   # 1-network world-model agent: real-env eval rewards
    rollouts = 50

    for i in range(rollouts):
        print(f"=== Starting run {i+1}/{rollouts} ===")

        # Step 1: Collect data from the real Reacher-v5 environment
        print("Collecting data from Reacher-v5...")
        env = gym.make("Reacher-v5")
        ppo = PPO('MlpPolicy', env=env, verbose=0)
        states, actions, next_states, done_flags = collect_data(
            ppo, n_episodes=2000, max_steps=50
        )

        # Step 2: Train the separate predictors and the unified world-model
        dl = prepare_dataloader(states, actions, next_states, batch_size=64)
        state_pred = StatePredictor(state_dim=10, action_dim=2)
        reward_pred = RewardPredictor(state_dim=10, action_dim=2)
        one_net_wm  = OneNetworkWM(state_dim=10, action_dim=2)

        print("\nTraining the state-prediction network...")
        train_predictor(state_pred, dl, epochs=20, lr=1e-3)

        dl_reward = prepare_dataloader_reward(states, actions, next_states, done_flags, batch_size=64)
        print("\nTraining the reward-prediction network...")
        train_reward_predictor(reward_pred, dl_reward, epochs=10, lr=1e-3)

        print("\nTraining the unified world-model network...")
        train_one_net_wm_model(one_net_wm, dl_reward, epochs=20, lr=1e-3)

        # Step 3: (Optional) Evaluate the state predictor on real episodes
        print("\nEvaluating the state predictor (optional)...")
        # evaluate_predictor(state_pred, ppo, n_episodes=100)

        # Step 4: Train and evaluate PPO agents in both world models
        print("\nTraining and evaluating world-model agents...")
        timesteps_record, rewards_wm3, rewards_wm1 = new_train_agent_in_world_model_with_eval(
            state_pred, reward_pred, one_net_wm,
            total_timesteps=40000,
            eval_interval=2000,
            num_eval_episodes=50
        )

        all_wm3_real_eval_rewards.append(rewards_wm3)
        all_wm1_real_eval_rewards.append(rewards_wm1)

        wm3_array = np.array(all_wm3_real_eval_rewards)
        wm1_array = np.array(all_wm1_real_eval_rewards)

        if i > 0:
            def compute_mean_ci(data, z=1.96):
                means = np.mean(data, axis=0)
                sem   = np.std(data, axis=0) / np.sqrt(data.shape[0])
                ci    = z * sem
                return means, ci

            mean_wm3, ci_wm3 = compute_mean_ci(wm3_array)
            mean_wm1, ci_wm1 = compute_mean_ci(wm1_array)

            plt.figure(figsize=(10, 6))
            plt.plot(timesteps_record, mean_wm3, '-', color="cyan",
                     label="3-Network WM Agent – Eval in Real Env (50 episodes)")
            plt.fill_between(timesteps_record,
                             mean_wm3 - ci_wm3,
                             mean_wm3 + ci_wm3,
                             color="cyan", alpha=0.2)

            plt.plot(timesteps_record, mean_wm1, '-', color="orange",
                     label="1-Network WM Agent – Eval in Real Env (50 episodes)")
            plt.fill_between(timesteps_record,
                             mean_wm1 - ci_wm1,
                             mean_wm1 + ci_wm1,
                             color="orange", alpha=0.2)

            plt.xlabel("Training Timesteps")
            plt.ylabel("Mean Episode Reward")
            plt.title(f"Reacher Mean Reward vs. World‐Model Network Count (run {i+1})")
            plt.legend()
            plt.grid(True)
            plt.savefig("number_network_comparison.png")
            plt.show()
            plt.close()



if __name__ == "__main__":
    main()

