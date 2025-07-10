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
    Runs the Reacher-v5 environment using the given policy and stores
    transitions: (state, action, next_state, reward).
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
# 2. State Prediction Network with PyTorch
# =====================================
class StatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Takes the current state (dim=state_dim) and an action vector
        (dim=action_dim) and predicts the next state (dim=state_dim).
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
        x = torch.cat([state.float(), action], dim=1)
        return state + self.fc(x)


def prepare_dataloader(states, actions, next_states, batch_size=32):
    """
    Prepares a DataLoader from the collected data.
    """
    states_t = torch.tensor(states, dtype=torch.float64)
    actions_t = torch.tensor(actions, dtype=torch.float32)
    next_states_t = torch.tensor(next_states, dtype=torch.float64)
    dataset = TensorDataset(states_t, actions_t, next_states_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_predictor(model, dataloader, epochs=20, lr=1e-3):
    """
    Trains the network using MSELoss to minimize the difference between
    the true next state and the predicted next state.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        for state_batch, action_batch, next_state_batch in dataloader:
            optimizer.zero_grad()
            pred_next = model(state_batch, action_batch)
            loss = criterion(pred_next, next_state_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * state_batch.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


def evaluate_predictor(model, ppo, n_episodes=10, max_steps=50):
    """
    Evaluates the network on actual Reacher-v5 episodes, computing
    MSE per step. Reports average per episode and overall.
    """
    env = gym.make("Reacher-v5")
    all_errors = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_errors = []
        steps = 0
        while not done and steps < max_steps:
            state_t = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
            action, _ = ppo.predict(state)
            action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            model.eval()
            with torch.no_grad():
                pred_next = model(state_t, action_t).squeeze(0).numpy()
            true_next, reward, done, truncated, info = env.step(action)
            error = np.mean((pred_next - true_next) ** 2)
            ep_errors.append(error)
            state = true_next
            steps += 1
        all_errors.append(np.mean(ep_errors))
    env.close()
    print("Overall average error over",
          n_episodes, "episodes:", np.mean(all_errors))


# ====================================
# 3. Reward Prediction Network
# ====================================
class RewardPredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Takes current state (dim=state_dim), action vector (dim=action_dim),
        and next state (dim=state_dim), and predicts a scalar reward.
        """
        super(RewardPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state, action, next_state):
        x = torch.cat([state.float(), action, next_state.float()], dim=1)
        return self.fc(x).squeeze(1)


def prepare_dataloader_reward(states, actions, next_states, rewards, batch_size=32):
    """
    Prepares a DataLoader for reward prediction.
    """
    states_t = torch.tensor(states, dtype=torch.float64)
    actions_t = torch.tensor(actions, dtype=torch.float32)
    next_states_t = torch.tensor(next_states, dtype=torch.float64)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    dataset = TensorDataset(states_t, actions_t, next_states_t, rewards_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_reward_predictor(model, dataloader, epochs=20, lr=1e-3):
    """
    Trains the reward prediction network using MSELoss to minimize the
    difference between predicted and true reward.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        for state_batch, action_batch, next_state_batch, reward_batch in dataloader:
            optimizer.zero_grad()
            pred_r = model(state_batch, action_batch, next_state_batch)
            loss = criterion(pred_r, reward_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * state_batch.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# =====================================
# 4. Define the world-model environment using the predictor network
# =====================================
class WorldModelReacherEnv(gym.Env):
    """
    Simulated environment (world model) that uses the predictor network to evolve the state.
    Reward and termination logic follow the same parameters as in CartPole-v1.
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
        self.total_reward = 0.0  # Reset total reward (if tracking)
        self.state = np.zeros(self.state_dim)  # Initial state (e.g., balanced)
        return self.state, {}

    def step(self, action):
        # Ensure action is a float tensor
        self.steps += 1
        state_tensor = torch.tensor(self.state, dtype=torch.float64).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

        self.state_predictor.eval()
        self.reward_predictor.eval()
        with torch.no_grad():
            next_state = self.state_predictor(state_tensor, action_tensor).squeeze(0).numpy()
            next_state_tensor = torch.tensor(next_state, dtype=torch.float64).unsqueeze(0)
            reward = self.reward_predictor(state_tensor, action_tensor, next_state_tensor).squeeze(0).item()

        self.state = next_state
        self.total_reward += reward

        terminated = False
        truncated = (self.steps >= self.max_steps)
        info = {}
        if terminated or truncated:
            # You can include episode length and accumulated reward in info as before
            info["episode"] = {"l": self.steps, "r": self.total_reward}

        return self.state, reward, terminated, truncated, info


# =====================================
# 5. Train the PPO agent in the world model
# =====================================
def train_agent_in_world_model(state_predictor, reward_predictor, timesteps=10000):
    """
    Train a PPO agent using stable-baselines3 on the simulated (world model)
    environment built from the predictor networks.
    """
    env = WorldModelReacherEnv(state_predictor, reward_predictor)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    env.close()
    return model


# =====================================
# 6. Test the trained policy in Reacher-v5 recording 5 episodes
# =====================================
def test_policy(model, n_episodes=5, video_folder="./videos"):
    """
    Runs the trained policy in the real Reacher-v5 environment.
    Uses the RecordVideo wrapper to record each episode.
    """
    os.makedirs(video_folder, exist_ok=True)
    env = gym.make("Reacher-v5", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda ep: True)

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(state)
            state, reward, done, terminated, info = env.step(action)
            total_reward += reward
        print(f"Test Episode {ep+1} - Total reward: {total_reward}")
    env.close()


from stable_baselines3.common.callbacks import BaseCallback

class EpisodeInfoCallback(BaseCallback):
    """
    Callback that extracts and stores the length (number of steps) of each episode.
    The Monitor wrapper adds, inside the `info` dict under the 'episode' key,
    a dict containing 'l' for episode length (in steps) and 'r' for total reward.
    """
    def __init__(self, verbose=0):
        super(EpisodeInfoCallback, self).__init__(verbose)
        self.episode_lengths = []   # list accumulating episode lengths
        self.last_values = 0

    def _on_step(self) -> bool:
        # Retrieve the "infos" list from the rollout data
        infos = self.locals.get("infos", [])
        for info in infos:
            # When 'episode' appears, an episode has just ended
            if "episode" in info:
                # We store the total reward ('r') or length ('l') as desired
                self.episode_lengths.append(info["episode"]["l"])
        return True

    def get_mean_episode_length(self):
        """
        Computes a running mean of episode lengths using:
          - the last up to 2,000 stored lengths
          - plus the previous batch of up to 100 lengths for smoothing
        """
        recent = self.episode_lengths[-2000:] if self.episode_lengths else []
        if self.last_values == 0 and len(self.episode_lengths) >= 100:
            self.last_values = self.episode_lengths[-100:]
        combined = recent + (self.last_values if isinstance(self.last_values, list) else [])
        mean_length = np.mean(combined) if combined else 0.0
        # Update last_values for the next call
        if len(self.episode_lengths) >= 100:
            self.last_values = self.episode_lengths[-100:]
        return mean_length

    def reset_episodes(self):
        """Clears stored episode lengths so only new episodes are counted."""
        self.episode_lengths = []


def new_train_agent_in_world_model_with_eval(
    state_predictor,
    end_predictor,
    total_timesteps=10000,
    eval_interval=2000,
    num_eval_episodes=50
):
    """
    Trains a PPO agent inside the learned world model, and every `eval_interval` timesteps:

      1. Records the agent’s average episode length during world-model training.
      2. Runs 50 evaluation episodes in the world model.
      3. Runs 50 evaluation episodes in the real Reacher-v5 environment.

    Simultaneously, trains a second PPO agent directly in the real environment,
    recording its average training and evaluation performance at the same intervals.

    Returns:
      wm_timesteps, wm_train_rewards, wm_eval_world, wm_eval_real,
      real_train_rewards, real_eval_rewards
    """

    # ==== Train World-Model Agent ====
    wm_env = Monitor(WorldModelReacherEnv(state_predictor, end_predictor))
    wm_agent = PPO("MlpPolicy", wm_env, verbose=0)
    wm_callback = EpisodeInfoCallback()

    wm_timesteps = []
    wm_train_rewards = []
    wm_eval_world = []
    wm_eval_real = []

    accumulated_steps = 0
    while accumulated_steps < total_timesteps:
        wm_agent.learn(
            total_timesteps=eval_interval,
            reset_num_timesteps=False,
            callback=wm_callback
        )
        accumulated_steps += eval_interval

        # 1) training metric
        avg_train_len = wm_callback.get_mean_episode_length()
        wm_timesteps.append(accumulated_steps)
        wm_train_rewards.append(avg_train_len)

        # 2) evaluate in world model
        world_rewards = []
        eval_env_wm = WorldModelReacherEnv(state_predictor, end_predictor)
        for _ in range(num_eval_episodes):
            ep_len = 0
            state, _ = eval_env_wm.reset()
            done = False
            while not done:
                action, _ = wm_agent.predict(state)
                state, _, done, truncated, _ = eval_env_wm.step(action)
                ep_len += 1
                if truncated:
                    break
            world_rewards.append(ep_len)
        eval_env_wm.close()
        wm_eval_world.append(np.mean(world_rewards))

        # 3) evaluate in real env
        real_rewards = []
        eval_env_real = Monitor(gym.make("Reacher-v5"))
        for _ in range(num_eval_episodes):
            ep_len = 0
            state, _ = eval_env_real.reset()
            done = False
            while not done:
                action, _ = wm_agent.predict(state)
                state, _, done, truncated, _ = eval_env_real.step(action)
                ep_len += 1
                if truncated:
                    break
            real_rewards.append(ep_len)
        eval_env_real.close()
        wm_eval_real.append(np.mean(real_rewards))

        print(f"[World-Model Agent] Steps: {accumulated_steps} | "
              f"TrainLen: {avg_train_len:.2f} | "
              f"EvalWorld: {wm_eval_world[-1]:.2f} | "
              f"EvalReal: {wm_eval_real[-1]:.2f}")

    wm_env.close()

    # ==== Train Real-Environment Agent ====
    real_env = Monitor(gym.make("Reacher-v5"))
    real_agent = PPO("MlpPolicy", real_env, verbose=0)
    real_callback = EpisodeInfoCallback()

    real_timesteps = []
    real_train_rewards = []
    real_eval_rewards = []

    real_steps = 0
    while real_steps < total_timesteps:
        real_agent.learn(
            total_timesteps=eval_interval,
            reset_num_timesteps=False,
            callback=real_callback
        )
        real_steps += eval_interval

        avg_real_train = real_callback.get_mean_episode_length()
        real_timesteps.append(real_steps)
        real_train_rewards.append(avg_real_train)

        # evaluation in real env
        eval_env = Monitor(gym.make("Reacher-v5"))
        eval_lengths = []
        for _ in range(num_eval_episodes):
            ep_len = 0
            state, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = real_agent.predict(state)
                state, _, done, truncated, _ = eval_env.step(action)
                ep_len += 1
                if truncated:
                    break
            eval_lengths.append(ep_len)
        eval_env.close()
        real_eval_rewards.append(np.mean(eval_lengths))

        print(f"[Real-Env Agent] Steps: {real_steps} | "
              f"TrainLen: {avg_real_train:.2f} | "
              f"EvalLen: {real_eval_rewards[-1]:.2f}")

    real_env.close()

    return (
        wm_timesteps,
        wm_train_rewards,
        wm_eval_world,
        wm_eval_real,
        real_train_rewards,
        real_eval_rewards
    )

# =====================================
# Main function
# =====================================
def main():
    all_wm_training_rewards  = []  # world-model training rewards (callback)
    all_wm_sim_world_rewards = []  # world-model simulated rewards: 50 episodes in world model
    all_wm_sim_real_rewards  = []  # world-model simulated rewards: 50 episodes in real env
    all_real_training_rewards = [] # real-env training rewards
    all_real_eval_rewards     = [] # real-env evaluation rewards

    rollouts = 50

    for i in range(rollouts):
        print(f"=== Starting training run {i+1}/{rollouts} ===")

        # Step 1: Collect data
        print("Collecting data from Reacher-v5...")
        env = gym.make("Reacher-v5")
        ppo = PPO('MlpPolicy', env=env, verbose=0)
        states, actions, next_states, rewards = collect_data(
            ppo, n_episodes=2000, max_steps=50
        )

        # Step 2: Prepare DataLoader and train prediction networks
        dataloader = prepare_dataloader(states, actions, next_states, batch_size=64)
        state_predictor  = StatePredictor(state_dim=10, action_dim=2)
        reward_predictor = RewardPredictor(state_dim=10, action_dim=2)
        print("\nTraining the state-prediction network...")
        train_predictor(state_predictor, dataloader, epochs=20, lr=1e-3)

        dataloader = prepare_dataloader_reward(states, actions, next_states, rewards, batch_size=64)
        print("\nTraining the reward-prediction network...")
        train_reward_predictor(reward_predictor, dataloader, epochs=10, lr=1e-3)

        # Step 3: Evaluate the state predictor on real episodes
        print("\nEvaluating the state predictor...")
        # evaluate_predictor(state_predictor, ppo, n_episodes=100)

        # Step 4: Train PPO agent in the simulated world-model
        print("\nTraining PPO agent in the world model...")
        (timesteps_record,
         wm_training_rewards,
         wm_sim_world_rewards,
         wm_sim_real_rewards,
         real_training_rewards,
         real_eval_rewards) = new_train_agent_in_world_model_with_eval(
             state_predictor,
             reward_predictor,
             total_timesteps=40000,
             eval_interval=2000,
             num_eval_episodes=50
        )

        # Only keep runs where the world-model agent performs above a threshold
        if (np.mean(wm_sim_world_rewards) >= -100
            or np.mean(wm_training_rewards) >= -100):
            all_wm_training_rewards.append(wm_training_rewards)
            all_wm_sim_world_rewards.append(wm_sim_world_rewards)
            all_wm_sim_real_rewards.append(wm_sim_real_rewards)
            all_real_training_rewards.append(real_training_rewards)
            all_real_eval_rewards.append(real_eval_rewards)

        # Stack into arrays for plotting
        plot_wm_training_rewards   = np.array(all_wm_training_rewards)
        plot_wm_sim_world_rewards  = np.array(all_wm_sim_world_rewards)
        plot_wm_sim_real_rewards   = np.array(all_wm_sim_real_rewards)
        plot_real_training_rewards = np.array(all_real_training_rewards)
        plot_real_eval_rewards     = np.array(all_real_eval_rewards)

        if i > 0:
            # Compute mean and 95% confidence interval for each eval point
            def compute_mean_ci(data, confidence_factor=1.96):
                means = np.mean(data, axis=0)
                stds  = np.std(data, axis=0)
                sem   = stds / np.sqrt(data.shape[0])
                ci    = confidence_factor * sem
                return means, ci

            wm_train_mean, wm_train_ci       = compute_mean_ci(plot_wm_training_rewards)
            wm_world_mean, wm_world_ci       = compute_mean_ci(plot_wm_sim_world_rewards)
            wm_real_mean,  wm_real_ci        = compute_mean_ci(plot_wm_sim_real_rewards)
            real_train_mean, real_train_ci   = compute_mean_ci(plot_real_training_rewards)
            real_eval_mean, real_eval_ci     = compute_mean_ci(plot_real_eval_rewards)

            # Plot evaluation comparison
            plt.figure(figsize=(10, 6))
            plt.plot(timesteps_record, wm_world_mean, '-', color="orange",
                     label="WM Agent – Eval in World Model (50 eps.)")
            plt.fill_between(timesteps_record,
                             wm_world_mean - wm_world_ci,
                             wm_world_mean + wm_world_ci,
                             color="orange", alpha=0.2)
            plt.plot(timesteps_record, wm_real_mean, '-', color="green",
                     label="WM Agent – Eval in Real Env (50 eps.)")
            plt.fill_between(timesteps_record,
                             wm_real_mean - wm_real_ci,
                             wm_real_mean + wm_real_ci,
                             color="green", alpha=0.2)
            plt.plot(timesteps_record, real_eval_mean, '-', color="purple",
                     label="Real Agent Eval Reward (50 eps.)")
            plt.fill_between(timesteps_record,
                             real_eval_mean - real_eval_ci,
                             real_eval_mean + real_eval_ci,
                             color="purple", alpha=0.2)
            plt.xlabel("Training Timesteps")
            plt.ylabel("Mean Episode Reward")
            plt.title(f"Reacher Mean Reward Comparison during Evaluation (run {i+1})")
            plt.legend(); plt.grid(True)
            plt.savefig("evaluation_reward_comparison.png")
            plt.show(); plt.close()

            # Plot training comparison
            plt.figure(figsize=(10, 6))
            plt.plot(timesteps_record, wm_train_mean, '-', color="blue",
                     label="World Model Training Reward")
            plt.fill_between(timesteps_record,
                             wm_train_mean - wm_train_ci,
                             wm_train_mean + wm_train_ci,
                             color="blue", alpha=0.2)
            plt.plot(timesteps_record, real_train_mean, '-', color="red",
                     label="Real Agent Training Reward")
            plt.fill_between(timesteps_record,
                             real_train_mean - real_train_ci,
                             real_train_mean + real_train_ci,
                             color="red", alpha=0.2)
            plt.xlabel("Training Timesteps")
            plt.ylabel("Mean Episode Reward")
            plt.title(f"Reacher Mean Reward Comparison during Training (run {i+1})")
            plt.legend(); plt.grid(True)
            plt.savefig("training_reward_comparison.png")
            plt.show(); plt.close()

if __name__ == "__main__":
    main()

