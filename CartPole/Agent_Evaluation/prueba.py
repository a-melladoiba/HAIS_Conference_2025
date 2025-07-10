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
# 1. CartPole-v1 data collection
# =====================================
def collect_data(ppo, n_episodes=100, max_steps=500):
    
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
# 2. State prediction network
# =====================================
class StatePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        
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
        
        action_onehot = torch.zeros((action.size(0), 2), device=action.device)
        action_onehot.scatter_(1, action.view(-1, 1), 1.)
        x = torch.cat([state, action_onehot], dim=1)
        return state + self.fc(x)

def prepare_dataloader(states, actions, next_states, dones, batch_size=32):
    
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.long)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
    dones_tensor = torch.tensor(dones, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(states_tensor, actions_tensor, next_states_tensor, dones_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_predictor(model, dataloader, epochs=20, lr=1e-3):
    
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
    
    env = gym.make("CartPole-v1")
    total_errors = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_errors = []
        steps = 0
        while not done and steps < max_steps:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
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
    env.close()
    print("Mean global error in ", n_episodes, "episodes:", np.mean(total_errors))

# ====================================
# 3. Termination prediction network
# ====================================
class TerminationPredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        
        super(TerminationPredictor, self).__init__()
        self.action_dim = action_dim  
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state, action, next_state):
        
        action_onehot = torch.zeros((action.size(0), self.action_dim), device=action.device)
        action_onehot.scatter_(1, action.view(-1, 1), 1.)
        
        x = torch.cat([state, action_onehot, next_state], dim=1)
        logits = self.fc(x)
        
        return torch.sigmoid(logits)

def train_termination_predictor(model, dataloader, epochs=20, lr=1e-3):
    
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
# 4. World Model
# =====================================
class WorldModelCartPoleEnv(gym.Env):
    
    metadata = {"render_modes": []}
    def __init__(self, state_predictor, end_predictor, state_dim=4):
        super(WorldModelCartPoleEnv, self).__init__()
        self.state_predictor = state_predictor
        self.end_predictor = end_predictor
        self.state_dim = state_dim
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.steps = 0
        self.max_steps = 500

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.total_reward = 0  
        self.state = np.zeros(self.state_dim)  
        return self.state, {}

    def step(self, action):
        
        action = int(action)
        self.steps += 1
        state_tensor = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.long)
        self.state_predictor.eval()
        self.end_predictor.eval()
        with torch.no_grad():
            next_state = self.state_predictor(state_tensor, action_tensor).squeeze(0).numpy()
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            terminated = bool(0.5 <= self.end_predictor(state_tensor, action_tensor, next_state_tensor))
        self.state = next_state

        reward = 1.0  
        self.total_reward += reward

        truncated = self.steps >= self.max_steps

        info = {}
        if terminated or truncated:
            
            info["episode"] = {"l": self.steps, "r": self.total_reward}
        return self.state, reward, terminated, truncated, info

# =====================================
# 5. PPO training in World Model
# =====================================

from stable_baselines3.common.callbacks import BaseCallback

class EpisodeInfoCallback(BaseCallback):
    
    def __init__(self, verbose=0):
        super(EpisodeInfoCallback, self).__init__(verbose)
        self.episode_lengths = []  
        self.last_values = 0

    def _on_step(self) -> bool:
        
        infos = self.locals.get("infos", [])
        for info in infos:
            
            if "episode" in info:
                self.episode_lengths.append(info["episode"]["l"])
        return True

    def get_mean_episode_length(self):
        new_mean = self.episode_lengths[-2000:] if len(self.episode_lengths) > 0 else 0.0
        
        if self.last_values == 0:
            self.last_values = self.episode_lengths[-100:]
        self.mean_episode_length = np.mean((new_mean + self.last_values))
        self.last_values = self.episode_lengths[-100:]
        return self.mean_episode_length

    def reset_episodes(self):
        self.episode_lengths = []


def new_train_agent_in_world_model_with_eval(state_predictor, end_predictor, total_timesteps=10000, eval_interval=2000, num_eval_episodes=50):

    # ----------------------------
    # Agent training in WM
    # ----------------------------
    training_env = Monitor(WorldModelCartPoleEnv(state_predictor, end_predictor))
    model = PPO("MlpPolicy", training_env, verbose=0)
    episode_callback = EpisodeInfoCallback()

    wm_timesteps_record = []       
    wm_training_rewards = []       
    wm_sim_world_rewards = []      
    wm_sim_real_rewards = []       

    total_evaluated_timesteps = 0

    while total_evaluated_timesteps < total_timesteps:
        
        model.learn(total_timesteps=eval_interval, reset_num_timesteps=False, callback=episode_callback)
        current_training_reward = episode_callback.get_mean_episode_length()
        total_evaluated_timesteps += eval_interval

        wm_timesteps_record.append(total_evaluated_timesteps)
        wm_training_rewards.append(current_training_reward)

        # --- Evaluation in WM (50 episodes) ---
        sim_wm_rewards = []
        wm_eval_env = WorldModelCartPoleEnv(state_predictor, end_predictor)
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

        # --- Eavluation in real-world (50 episodes) for agent trained in WM ---
        sim_real_rewards = []
        real_eval_env = gym.make("CartPole-v1")
        real_eval_env = Monitor(real_eval_env)
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

        print(f"[World Model Agent] Total Steps: {total_evaluated_timesteps} -> "
              f"Training Reward: {current_training_reward:.2f}, "
              f"WorldModel Eval (50 epis.) Reward: {wm_sim_world_rewards[-1]:.2f}, "
              f"Real Env Eval (50 epis.) Reward: {wm_sim_real_rewards[-1]:.2f}")

    training_env.close()

    # ----------------------------
    # Real-world agent training
    # ----------------------------
    real_training_env = gym.make("CartPole-v1")
    real_training_env = Monitor(real_training_env)
    real_agent = PPO("MlpPolicy", real_training_env, verbose=0)
    episode_callback_real = EpisodeInfoCallback()
    
    real_timesteps_record = []    
    real_training_rewards = []    
    real_eval_rewards = []        

    total_evaluated_real_timesteps = 0

    while total_evaluated_real_timesteps < total_timesteps:
        real_agent.learn(total_timesteps=eval_interval, reset_num_timesteps=False, callback=episode_callback_real)
        current_real_reward = episode_callback_real.get_mean_episode_length()
        total_evaluated_real_timesteps += eval_interval

        real_timesteps_record.append(total_evaluated_real_timesteps)
        real_training_rewards.append(current_real_reward)
        
        # --- Evaluation in real-world (50 episodes) for real-world trained agent ---
        sim_real_agent_rewards = []
        real_eval_env = gym.make("CartPole-v1")
        real_eval_env = Monitor(real_eval_env)
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

        print(f"[Real Agent] Total Steps: {total_evaluated_real_timesteps} -> "
              f"Training Reward: {current_real_reward:.2f}, Eval Reward (50 epis.): {mean_eval_reward:.2f}")

    real_training_env.close()

    return wm_timesteps_record, wm_training_rewards, wm_sim_world_rewards, wm_sim_real_rewards, real_training_rewards, real_eval_rewards


# =====================================
# Main function
# =====================================
def main():
    all_wm_training_rewards  = []  
    all_wm_sim_world_rewards = []  
    all_wm_sim_real_rewards  = []  
    all_real_training_rewards = []
    all_real_eval_rewards = []

    rollouts = 50

    for i in range(rollouts):
        print(f"=== Rollout {i+1}/{rollouts} ===")
        # Step 1
        print("Collecting data form CartPole-v1...")
        env = gym.make("CartPole-v1")
        ppo = PPO('MlpPolicy', env=env, verbose=0)
        
        estados, acciones, siguientes_estados, terminaciones = collect_data(ppo, n_episodes=2000, max_steps=500)
        
        # Step 2
        dataloader = prepare_dataloader(estados, acciones, siguientes_estados, terminaciones, batch_size=64)
        state_predictor = StatePredictor(state_dim=4, action_dim=2)
        termination_predictor = TerminationPredictor(state_dim=4, action_dim=2)
        print("\nTraining state prediction network...")
        train_predictor(state_predictor, dataloader, epochs=10, lr=1e-3)
        
        print("\nTraining termination prediction network...")
        train_termination_predictor(termination_predictor, dataloader, epochs=50, lr=1e-3)
        
        # Step 3
        print("\nEvaluating prediction network...")
        #evaluate_predictor(state_predictor, ppo, n_episodes=100)
        
        # Step 4
        print("\nTraining PPO in World Model...")
        timesteps_record, wm_training_rewards, wm_sim_world_rewards, wm_sim_real_rewards, real_training_rewards, real_eval_rewards = new_train_agent_in_world_model_with_eval(state_predictor, termination_predictor, total_timesteps=60000, eval_interval=2000, num_eval_episodes=50)

        all_wm_training_rewards.append(wm_training_rewards)
        all_wm_sim_world_rewards.append(wm_sim_world_rewards)
        all_wm_sim_real_rewards.append(wm_sim_real_rewards)
        all_real_training_rewards.append(real_training_rewards)
        all_real_eval_rewards.append(real_eval_rewards)
    
        plot_wm_training_rewards  = np.array(all_wm_training_rewards)  
        plot_wm_sim_world_rewards = np.array(all_wm_sim_world_rewards)
        plot_wm_sim_real_rewards  = np.array(all_wm_sim_real_rewards)
        plot_real_training_rewards = np.array(all_real_training_rewards)
        plot_real_eval_rewards = np.array(all_real_eval_rewards)

        if i > 0:
            
            def compute_mean_ci(data, confidence_factor=1.96):
                means = np.mean(data, axis=0)
                stds  = np.std(data, axis=0)
                sem   = stds / np.sqrt(data.shape[0])
                ci    = confidence_factor * sem
                return means, ci

            wm_training_mean, wm_training_ci = compute_mean_ci(plot_wm_training_rewards)
            wm_sim_world_mean, wm_sim_world_ci = compute_mean_ci(plot_wm_sim_world_rewards)
            wm_sim_real_mean,  wm_sim_real_ci  = compute_mean_ci(plot_wm_sim_real_rewards)
            real_training_mean, real_training_ci = compute_mean_ci(plot_real_training_rewards)
            real_eval_mean, real_eval_ci = compute_mean_ci(plot_real_eval_rewards)

            
            plt.figure(figsize=(10, 6))
            plt.plot(timesteps_record, wm_sim_world_mean, '-', color="orange", label="WM Agent - Eval in World Model (50 epis.)")
            plt.fill_between(timesteps_record,
                        wm_sim_world_mean - wm_sim_world_ci,
                        wm_sim_world_mean + wm_sim_world_ci,
                        color="orange", alpha=0.2)
            plt.plot(timesteps_record, wm_sim_real_mean, '-', color="green", label="WM Agent - Eval in Real Env (50 epis.)")
            plt.fill_between(timesteps_record,
                        wm_sim_real_mean - wm_sim_real_ci,
                        wm_sim_real_mean + wm_sim_real_ci,
                        color="green", alpha=0.2)
            plt.plot(timesteps_record, real_eval_mean, '-', color="purple", label="Real Agent Evaluation Reward (50 epis.)")
            plt.fill_between(timesteps_record,
                        real_eval_mean - real_eval_ci,
                        real_eval_mean + real_eval_ci,
                        color="purple", alpha=0.2)
            plt.xlabel("Training Timesteps")
            plt.ylabel("Mean Episode Reward")
            plt.title(f"CartPole Mean Reward Comparison during Agent Evaluation ({i+1} rollouts)")
            plt.legend()
            plt.grid(True)
            plt.savefig("evaluation_reward_comparison.png")
            plt.show()
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(timesteps_record, wm_training_mean, '-', color="blue", label="World Model Training Reward")
            plt.fill_between(timesteps_record,
                        wm_training_mean - wm_training_ci,
                        wm_training_mean + wm_training_ci,
                        color="blue", alpha=0.2)
            plt.plot(timesteps_record, real_training_mean, '-', color="red", label="Real Agent Training Reward")
            plt.fill_between(timesteps_record,
                        real_training_mean - real_training_ci,
                        real_training_mean + real_training_ci,
                        color="red", alpha=0.2)
            plt.xlabel("Training Timesteps")
            plt.ylabel("Mean Episode Reward")
            plt.title(f"CartPole Mean Reward Comparison during Agent Training ({i+1} rollouts)")
            plt.legend()
            plt.grid(True)
            plt.savefig("training_reward_comparison.png")
            plt.show()
            plt.close()

if __name__ == "__main__":
    main()

