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
# 1. Data collection in CartPole-v1
# =====================================
def collect_data(ppo, n_episodes=100, max_steps=500):
    """
    Runs through the CartPole-v1 environment using a random policy and stores
    transitions: (state, action, next state, done).
    """
    env = gym.make("CartPole-v1")
    estados = []
    acciones = []
    siguientes_estados = []
    dones = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        for t in range(max_steps):
            if done:
                break
            action, _ = ppo.predict(state)
            next_state, reward, done, truncated, info = env.step(action)
            estados.append(state)
            acciones.append(action)
            siguientes_estados.append(next_state)
            dones.append(done)
            state = next_state
    env.close()
    return np.array(estados), np.array(acciones), np.array(siguientes_estados), np.array(dones)


# =====================================
# 2. State prediction network with PyTorch
# =====================================
class StatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        The network takes the current state (dim=state_dim) and the action in one-hot format (dim=action_dim),
        and predicts the next state (dim=state_dim).
        """
        super(StatePredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SELU(),
            nn.LayerNorm(hidden_dim//2),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim//2, state_dim)
        )
    
    def forward(self, state, action):
        # Convert action (integer) to one-hot vector
        action_onehot = torch.zeros((action.size(0), 2), device=action.device)
        action_onehot.scatter_(1, action.view(-1, 1), 1.)
        x = torch.cat([state, action_onehot], dim=1)
        return state + self.fc(x)


def prepare_dataloader(estados, acciones, siguientes_estados, terminaciones, batch_size=32):
    """
    Prepares a DataLoader from the collected data.
    """
    estados_tensor = torch.tensor(estados, dtype=torch.float32)
    acciones_tensor = torch.tensor(acciones, dtype=torch.long)
    siguientes_estados_tensor = torch.tensor(siguientes_estados, dtype=torch.float32)
    terminaciones_tensor = torch.tensor(terminaciones, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(estados_tensor, acciones_tensor, siguientes_estados_tensor, terminaciones_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_predictor(model, dataloader, epochs=20, lr=1e-3):
    """
    Trains the network using MSELoss to minimize the difference between the real
    next state and the predicted one.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for batch in dataloader:
            state_batch, action_batch, next_state_batch, termination_batch = batch
            optimizer.zero_grad()
            pred_next_state = model(state_batch, action_batch)
            loss = criterion(pred_next_state, next_state_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * state_batch.size(0)
        epoch_loss /= len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")


def evaluate_predictor(model, ppo, n_episodes=10, max_steps=500):
    """
    Evaluates the network on real episodes of CartPole-v1, computing the error (MSE)
    in the prediction at each step. Reports the average error per episode and overall.
    """
    env = gym.make("CartPole-v1")
    total_errors = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_errors = []
        steps = 0
        while not done and steps < max_steps:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # Take a random action for evaluation
            action, _ = ppo.predict(state)
            action_tensor = torch.tensor([int(action)], dtype=torch.long)
            model.eval()
            with torch.no_grad():
                pred_next_state = model(state_tensor, action_tensor).squeeze(0).numpy()
            true_next_state, reward, done, truncated, info = env.step(action)
            error = np.mean((pred_next_state - true_next_state) ** 2)
            episode_errors.append(error)
            state = true_next_state
            steps += 1
        err_media = np.mean(episode_errors)
        total_errors.append(err_media)
        # print(f"Episode {ep+1} - Average prediction error: {err_media:.4f}")
    env.close()
    print("Overall average error over", n_episodes, "episodes:", np.mean(total_errors))


# ====================================
# 3. Termination prediction network
# ====================================
class TerminationPredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        The network takes the current state (dim=state_dim), the action in one-hot format (dim=action_dim),
        and the next state (dim=state_dim), and predicts the termination probability (scalar between 0 and 1).
        """
        super(TerminationPredictor, self).__init__()
        self.action_dim = action_dim  # Store action dimension for one-hot
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state, action, next_state):
        # Convert action (integer) to one-hot vector with dimension action_dim
        action_onehot = torch.zeros((action.size(0), self.action_dim), device=action.device)
        action_onehot.scatter_(1, action.view(-1, 1), 1.)
        # Concatenate current state, one-hot action, and next state
        x = torch.cat([state, action_onehot, next_state], dim=1)
        logits = self.fc(x)
        # Apply sigmoid to obtain a probability
        return torch.sigmoid(logits)


def train_termination_predictor(model, dataloader, epochs=20, lr=1e-3):
    """
    Trains the termination prediction network using BCELoss to minimize the difference
    between the predicted probability and the true label (0 or 1).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for batch in dataloader:
            state_batch, action_batch, next_state_batch, term_batch = batch
            optimizer.zero_grad()
            pred_term = model(state_batch, action_batch, next_state_batch)
            loss = criterion(pred_term, term_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * state_batch.size(0)
        
        epoch_loss /= len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# =====================================
#   World Model Single Network
# =====================================
class OneNetworkWM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        The network takes as input the current state (dim=state_dim) and the action in one-hot format (dim=action_dim),
        and predicts:
            - The next state: current state + estimated delta.
            - The step reward: a scalar value.
            - Whether the episode ends: probability (done flag).
        """
        super(OneNetworkWM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared layer that processes the concatenated state+action
        self.shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.05)
        )
        
        # Head to predict the state delta (to obtain the next state)
        self.state_head = nn.Linear(hidden_dim // 2, state_dim)
        # Head to predict if the episode ends (use sigmoid for a 0–1 value)
        self.done_head = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, state, action):
        # Convert action (integer) to one-hot vector
        batch_size = action.size(0)
        action_onehot = torch.zeros(batch_size, self.action_dim, device=action.device)
        action_onehot.scatter_(1, action.view(-1, 1), 1.)
        
        # Concatenate state and one-hot action
        x = torch.cat([state, action_onehot], dim=1)
        features = self.shared(x)
        
        # Predict state delta (added to state to get next state)
        delta_state = self.state_head(features)
        next_state = state + delta_state
    
        # Predict episode termination (done flag)
        done_logits = self.done_head(features)
        done_prob = torch.sigmoid(done_logits)  # interpreted as probability
        
        return next_state, done_prob

def train_one_net_wm_model(model, dataloader, epochs=20, lr=1e-3, device='cpu'):
    """
    Trains the integrated network which, from the state and the action (one-hot), predicts:
      - The next state (current state + estimated delta).
      - The step reward.
      - The probability that the episode ends.
      
    Uses:
      - MSELoss for state and reward.
      - BCELoss for the 'done' flag prediction.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss functions for each output:
    state_criterion = nn.MSELoss()
    done_criterion = nn.BCELoss()
    
    model.to(device)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for batch in dataloader:
            # Expect batch to include: state, action, next state, reward, and done
            state_batch, action_batch, next_state_batch, done_batch = batch
            
            # Move data to the device (CPU or GPU)
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            next_state_batch = next_state_batch.to(device)
            done_batch = done_batch.to(device)
            
            optimizer.zero_grad()
            
            # The network returns:
            # - pred_next_state: predicted next state (current + delta)
            # - pred_done: termination probability (0–1)
            pred_next_state, pred_done = model(state_batch, action_batch)
            
            # Compute loss for each head
            loss_state = state_criterion(pred_next_state, next_state_batch)
            loss_done = done_criterion(pred_done, done_batch.view(-1, 1).float())
            
            # Total loss: sum of all losses (weighting optional)
            loss = loss_state + loss_done
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * state_batch.size(0)
        
        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# =====================================
# 4. Define the world model environment using the predictor network
# =====================================
class WorldModelCartPoleEnv(gym.Env):
    """
    Simulated environment (world model) that uses the predictor network to evolve the state.
    The reward and termination condition are based on the same parameters as in CartPole-v1.
    """
    metadata = {"render_modes": []}
    def __init__(self, state_predictor, end_predictor, state_dim=4):
        super(WorldModelCartPoleEnv, self).__init__()
        self.state_predictor = state_predictor
        self.end_predictor = end_predictor
        self.state_dim = state_dim
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.steps = 0
        self.max_steps = 500

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.total_reward = 0  # accumulate reward if desired
        self.state = np.zeros(self.state_dim)  # initial state (e.g., balanced pole)
        return self.state, {}

    def step(self, action):
        # Ensure action is an integer
        action = int(action)
        self.steps += 1

        state_tensor = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.long)

        self.state_predictor.eval()
        self.end_predictor.eval()
        with torch.no_grad():
            next_state = (
                self.state_predictor(state_tensor, action_tensor)
                .squeeze(0)
                .numpy()
            )
            next_state_tensor = torch.tensor(
                next_state, dtype=torch.float32
            ).unsqueeze(0)
            # Termination probability ≥ 0.5 means done
            terminated = bool(
                0.5 <= self.end_predictor(
                    state_tensor, action_tensor, next_state_tensor
                )
            )

        self.state = next_state
        reward = 1.0  # constant reward per step
        self.total_reward += reward
        truncated = self.steps >= self.max_steps

        info = {}
        if terminated or truncated:
            # include episode length and total reward in info if desired
            info["episode"] = {"l": self.steps, "r": self.total_reward}

        return self.state, reward, terminated, truncated, info


#
#   New WM
#
class NewWorldModelCartPoleEnv(gym.Env):
    """
    Simulated environment (world model) that uses the world-model network to evolve the state.
    The reward and termination condition use the same parameters as in CartPole-v1.
    """
    metadata = {"render_modes": []}
    def __init__(self, wm_predictor, state_dim=4):
        super(NewWorldModelCartPoleEnv, self).__init__()
        self.wm_predictor = wm_predictor
        self.state_dim = state_dim
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.steps = 0
        self.max_steps = 500

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.total_reward = 0  # accumulate reward if desired
        self.state = np.zeros(self.state_dim)  # initial state (e.g., balanced pole)
        return self.state, {}

    def step(self, action):
        # Ensure action is an integer
        action = int(action)
        self.steps += 1

        state_tensor = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.long)

        with torch.no_grad():
            next_state, terminated = self.wm_predictor(state_tensor, action_tensor)
            next_state = next_state.squeeze(0).numpy()
            # Termination probability ≥ 0.5 means done
            terminated = bool(0.5 <= terminated)

        self.state = next_state
        reward = 1.0  # constant reward per step
        self.total_reward += reward
        truncated = self.steps >= self.max_steps

        info = {}
        if terminated or truncated:
            # include episode length and total reward in info if desired
            info["episode"] = {"l": self.steps, "r": self.total_reward}

        return self.state, reward, terminated, truncated, info

# =====================================
# 5. Training the PPO agent in the world model
# =====================================

from stable_baselines3.common.callbacks import BaseCallback

class EpisodeInfoCallback(BaseCallback):
    """
    Callback to extract and store the duration (number of steps) of each episode.
    The Monitor env sends, inside the info dict, an 'episode' dict
    containing, among other values, 'l' (duration in steps).
    """
    def __init__(self, verbose=0):
        super(EpisodeInfoCallback, self).__init__(verbose)
        self.episode_lengths = []  # list accumulating episode lengths
        self.last_values = 0

    def _on_step(self) -> bool:
        # Extract the "infos" field from rollouts
        infos = self.locals.get("infos", [])
        for info in infos:
            # If 'episode' in info, an episode has ended
            if "episode" in info:
                self.episode_lengths.append(info["episode"]["l"])
        return True

    def get_mean_episode_length(self):
        new_mean = self.episode_lengths[-2000:] if len(self.episode_lengths) > 0 else 0.0

        if self.last_values == 0:
            self.last_values = self.episode_lengths[-100:]
        self.mean_episode_length = np.mean(new_mean + self.last_values)
        self.last_values = self.episode_lengths[-100:]
        return self.mean_episode_length

    def reset_episodes(self):
        """Reset episode statistics so only recent steps are considered."""
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
    Trains a PPO agent in the world model environment and, every eval_interval steps,
    evaluates the current policy in three scenarios:
      1. Training performance in the world model (via callback statistics).
      2. Simulating num_eval_episodes in the real CartPole-v1 env with the trained agent.
      3. Training a fresh PPO agent directly in the real env, tracking its training reward
         and evaluating every eval_interval steps on num_eval_episodes.
    Metrics are collected and plotted in 'episode_length_comparison.png'.
    """

    # ----------------------------
    # Training the agent in the three-network world model
    # ----------------------------
    three_net_env = Monitor(WorldModelCartPoleEnv(state_predictor, end_predictor))
    three_net_model = PPO("MlpPolicy", three_net_env, verbose=0)
    episode_callback = EpisodeInfoCallback()

    wm_timesteps_record = []
    wm_sim_real_rewards = []
    total_evaluated_timesteps = 0

    while total_evaluated_timesteps < total_timesteps:
        # train for eval_interval steps in the world model
        three_net_model.learn(
            total_timesteps=eval_interval,
            reset_num_timesteps=False,
            callback=episode_callback
        )
        total_evaluated_timesteps += eval_interval
        wm_timesteps_record.append(total_evaluated_timesteps)

        # --- Evaluate in the real env (simulate num_eval_episodes) ---
        sim_real_rewards = []
        real_eval_env = Monitor(gym.make("CartPole-v1"))
        for _ in range(num_eval_episodes):
            ep_reward = 0
            state, _ = real_eval_env.reset()
            while True:
                action, _ = three_net_model.predict(state)
                state, reward, done, truncated, info = real_eval_env.step(action)
                ep_reward += reward
                if done or truncated:
                    break
            sim_real_rewards.append(ep_reward)
        wm_sim_real_rewards.append(np.mean(sim_real_rewards))
        real_eval_env.close()

        print(
            f"[3 Net World Model Agent] Total Steps: {total_evaluated_timesteps} -> "
            f"Real Env Eval (50 episodes) Reward: {wm_sim_real_rewards[-1]:.2f}"
        )

    # ----------------------------
    # Training the agent in the three-network world model
    # ----------------------------
    one_net_env = Monitor(NewWorldModelCartPoleEnv(wm_predictor))
    one_net_model = PPO("MlpPolicy", one_net_env, verbose=0)
    episode_callback = EpisodeInfoCallback()

    wm_1_timesteps_record = []
    wm_1_sim_real_rewards = []
    total_evaluated_timesteps = 0

    while total_evaluated_timesteps < total_timesteps:
        # train for eval_interval steps in the world model
        one_net_model.learn(
            total_timesteps=eval_interval,
            reset_num_timesteps=False,
            callback=episode_callback
        )
        total_evaluated_timesteps += eval_interval
        wm_1_timesteps_record.append(total_evaluated_timesteps)

        # --- Evaluate in the real env (simulate num_eval_episodes) ---
        sim_real_rewards = []
        real_eval_env = Monitor(gym.make("CartPole-v1"))
        for _ in range(num_eval_episodes):
            ep_reward = 0
            state, _ = real_eval_env.reset()
            while True:
                action, _ = one_net_model.predict(state)
                state, reward, done, truncated, info = real_eval_env.step(action)
                ep_reward += reward
                if done or truncated:
                    break
            sim_real_rewards.append(ep_reward)
        wm_1_sim_real_rewards.append(np.mean(sim_real_rewards))
        real_eval_env.close()

        print(
            f"[1 Net World Model Agent] Total Steps: {total_evaluated_timesteps} -> "
            f"Real Env Eval (50 episodes) Reward: {wm_1_sim_real_rewards[-1]:.2f}"
        )

    return wm_timesteps_record, wm_sim_real_rewards, wm_1_sim_real_rewards


# =====================================
# Main function
# =====================================
def main():
    all_wm_sim_real_rewards  = []  # Simulated WM: 50 episodes in real env
    all_wm_1_sim_real_rewards = []
    rollouts = 50

    for i in range(rollouts):
        print(f"=== Starting training {i+1}/{rollouts} ===")
        # Step 1: Data collection
        print("Collecting CartPole-v1 data...")
        env = gym.make("CartPole-v1")
        ppo = PPO('MlpPolicy', env=env, verbose=0)
        
        estados, acciones, siguientes_estados, terminaciones = collect_data(
            ppo, n_episodes=2000, max_steps=500
        )
        
        # Step 2: Prepare DataLoader and train predictors
        dataloader = prepare_dataloader(
            estados, acciones, siguientes_estados, terminaciones, batch_size=64
        )
        state_predictor = StatePredictor(state_dim=4, action_dim=2)
        termination_predictor = TerminationPredictor(state_dim=4, action_dim=2)
        one_net_wm = OneNetworkWM(state_dim=4, action_dim=2)

        print("\nTraining the state prediction network...")
        train_predictor(state_predictor, dataloader, epochs=10, lr=1e-3)
        print("\nTraining the termination prediction network...")
        train_termination_predictor(termination_predictor, dataloader, epochs=50, lr=1e-3)
        print("\nTraining the world model network...")
        train_one_net_wm_model(one_net_wm, dataloader, epochs=20, lr=1e-3)
        
        # Step 3: Evaluate the state predictor on real episodes
        print("\nEvaluating the predictor network...")
        # evaluate_predictor(state_predictor, ppo, n_episodes=100)
        
        # Step 4: Train a PPO agent in the simulated environment
        print("\nTraining the PPO agent in the world model...")
        timesteps_record, wm_sim_real_rewards, wm_1_sim_real_rewards = new_train_agent_in_world_model_with_eval(
            state_predictor,
            termination_predictor,
            one_net_wm,
            total_timesteps=60000,
            eval_interval=2000,
            num_eval_episodes=50
        )

        all_wm_sim_real_rewards.append(wm_sim_real_rewards)
        all_wm_1_sim_real_rewards.append(wm_1_sim_real_rewards)

        plot_wm_sim_real_rewards  = np.array(all_wm_sim_real_rewards)
        plot_wm_1_sim_real_rewards = np.array(all_wm_1_sim_real_rewards)

        if i > 0:
            # Function to compute mean and 95% confidence interval for each point
            def compute_mean_ci(data, confidence_factor=1.96):
                means = np.mean(data, axis=0)
                stds  = np.std(data, axis=0)
                sem   = stds / np.sqrt(data.shape[0])
                ci    = confidence_factor * sem
                return means, ci

            wm_sim_real_mean,  wm_sim_real_ci  = compute_mean_ci(plot_wm_sim_real_rewards)
            wm_1_sim_real_mean, wm_1_sim_real_ci = compute_mean_ci(plot_wm_1_sim_real_rewards)

            # Plot the two agents with error bars
            plt.figure(figsize=(10, 6))
            plt.plot(
                timesteps_record,
                wm_sim_real_mean,
                '-',
                color="cyan",
                label="Specialized-Networks WM Agent - Eval in Real Env (50 ep.)"
            )
            plt.fill_between(
                timesteps_record,
                wm_sim_real_mean - wm_sim_real_ci,
                wm_sim_real_mean + wm_sim_real_ci,
                color="cyan",
                alpha=0.2
            )
            plt.plot(
                timesteps_record,
                wm_1_sim_real_mean,
                '-',
                color="orange",
                label="Single-Network WM Agent - Eval in Real Env (50 ep.)"
            )
            plt.fill_between(
                timesteps_record,
                wm_1_sim_real_mean - wm_1_sim_real_ci,
                wm_1_sim_real_mean + wm_1_sim_real_ci,
                color="orange",
                alpha=0.2
            )
            plt.xlabel("Training Timesteps")
            plt.ylabel("Mean Episode Reward")
            plt.title(
                f"CartPole Mean Reward Comparison for Different Number of Networks in WM ({i+1} rollouts)"
            )
            plt.legend()
            plt.grid(True)
            plt.savefig("number_network_comparison.png")
            plt.show()
            plt.close()



if __name__ == "__main__":
    main()

