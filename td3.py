import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime

# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 32)
        self.layer_2 = nn.Linear(32, 16)
        self.layer_3 = nn.Linear(16, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.layer_1 = nn.Linear(state_dim + action_dim, 32)
        self.layer_2 = nn.Linear(32, 16)
        self.layer_3 = nn.Linear(16, 1)
        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(state_dim + action_dim, 32)
        self.layer_5 = nn.Linear(32, 16)
        self.layer_6 = nn.Linear(16, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Forward-Propagation on the second Critic Neural Network
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        print("\nInitializing TD3 Agent...")
        print(f"State Dimension: {state_dim}")
        print(f"Action Dimension: {action_dim}")
        print(f"Max Action: {max_action}")
        print(f"Using device: {device}")
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.max_action = max_action
        
        # Exploration parameters
        self.exploration_noise = 0.1
        self.exploration_decay = 0.995
        self.min_exploration = 0.01
        
        # Performance monitoring parameters
        self.best_reward = float('-inf')
        self.reward_history = []
        self.episode_lengths = []
        self.success_count = 0
        self.total_episodes = 0
        
        # Early stopping parameters
        self.patience = 10  # Number of evaluations to wait before early stopping
        self.patience_counter = 0
        self.min_improvement = 0.1  # Minimum improvement required to reset patience
        
        # Checkpointing parameters
        self.checkpoint_frequency = 50000  # Save checkpoint every 50000 timesteps
        self.last_checkpoint = 0
        
        print("\nExploration Parameters:")
        print(f"Initial Noise: {self.exploration_noise}")
        print(f"Decay Rate: {self.exploration_decay}")
        print(f"Minimum Noise: {self.min_exploration}")
        print("\nPerformance Monitoring Parameters:")
        print(f"Patience: {self.patience}")
        print(f"Minimum Improvement: {self.min_improvement}")
        print(f"Checkpoint Frequency: {self.checkpoint_frequency}")
        print("\nTD3 Agent initialized successfully!")

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        # Add exploration noise
        noise = np.random.normal(0, self.exploration_noise, size=action.shape)
        action = action + noise
        action = np.clip(action, -self.max_action, self.max_action)
        
        # Decay exploration noise
        old_noise = self.exploration_noise
        self.exploration_noise = max(self.min_exploration, 
                                   self.exploration_noise * self.exploration_decay)
        
        # Print exploration info every 1000 steps
        if hasattr(self, 'step_counter'):
            self.step_counter += 1
            if self.step_counter % 1000 == 0:
                print(f"\nExploration Update:")
                print(f"Previous Noise: {old_noise:.4f}, Current Noise: {self.exploration_noise:.4f}, Action: {action[0]:.4f}")
        else:
            self.step_counter = 0
        
        return action

    def train(self, replay_buffer, iterations, batch_size=200, discount=0.99, tau=0.005, 
              policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        if not hasattr(self, 'training_step'):
            self.training_step = 0
            print("\nStarting TD3 Training...")
            print(f"Batch Size: {batch_size}")
            print(f"Discount Factor: {discount}")
            print(f"Policy Update Frequency: {policy_freq}")
            print(f"Target Network Update Rate (tau): {tau}")
        
        self.training_step += 1
        
        # Check if we should save a checkpoint
        if self.training_step - self.last_checkpoint >= self.checkpoint_frequency:
            self.save(f"checkpoint_{self.training_step}", "./checkpoints")
            self.last_checkpoint = self.training_step
            print(f"\nSaved checkpoint at step {self.training_step}")
        
        for it in range(iterations):
            # Sample a batch of transitions from the replay buffer
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            
            # Convert to tensors and ensure correct dimensions
            state = torch.FloatTensor(batch_states).to(device)
            next_state = torch.FloatTensor(batch_next_states).to(device)
            action = torch.FloatTensor(batch_actions).squeeze(-1).to(device)
            reward = torch.FloatTensor(batch_rewards).to(device)
            done = torch.FloatTensor(batch_dones).to(device)

            # Select next action according to target policy
            next_action = self.actor_target(next_state)
            noise = torch.randn_like(next_action) * policy_noise
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q-values
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Compute current Q-values
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss with gradient clipping
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_optimizer.step()

                # Update target networks
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                # Print training metrics every 1000 steps
                if self.training_step % 1000 == 0:
                    print(f"\nTraining Metrics (Step {self.training_step}):")
                    print(f"Critic Loss: {critic_loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}, Average Q-Value: {(current_Q1.mean().item() + current_Q2.mean().item())/2:.4f}")

    def evaluate_policy(self, env, eval_episodes=10):
        """
        Evaluates the policy by calculating its average reward over a number of episodes.
        Also handles early stopping and performance monitoring.
        
        Args:
            env: The environment to evaluate in
            eval_episodes: Number of episodes to evaluate over
            
        Returns:
            float: Average reward over the evaluation episodes
        """
        print("\nStarting Policy Evaluation...")
        avg_reward = 0.
        success_rate = 0.
        avg_episode_length = 0.
        
        for episode in range(eval_episodes):
            print(f"\nEpisode {episode + 1}/{eval_episodes}")
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            success = False
            
            while not done:
                action = self.select_action(np.array(obs))
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                # Check if target was reached (success condition)
                if reward >= 5:  # Assuming 5 is the reward for reaching target
                    success = True
                
            avg_reward += episode_reward
            avg_episode_length += episode_steps
            if success:
                success_rate += 1
                
            print(f"Episode Reward: {episode_reward:.2f}")
            print(f"Episode Length: {episode_steps}")
            print(f"Success: {'Yes' if success else 'No'}")
            
        # Calculate averages
        avg_reward /= eval_episodes
        success_rate = (success_rate / eval_episodes) * 100
        avg_episode_length /= eval_episodes
        
        # Update performance history
        self.reward_history.append(avg_reward)
        self.episode_lengths.append(avg_episode_length)
        self.total_episodes += eval_episodes
        
        # Early stopping check
        if avg_reward > self.best_reward + self.min_improvement:
            self.best_reward = avg_reward
            self.patience_counter = 0
            # Save best model
            self.save("best_model", "./checkpoints")
        else:
            self.patience_counter += 1
            
        # Print evaluation summary
        print("\n" + "="*50)
        print("Evaluation Summary:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Episode Length: {avg_episode_length:.1f}")
        print(f"Best Reward So Far: {self.best_reward:.2f}")
        print(f"Patience Counter: {self.patience_counter}/{self.patience}")
        print("="*50 + "\n")
        
        # Check for early stopping
        if self.patience_counter >= self.patience:
            print("\nEarly stopping triggered!")
            print("No improvement in performance for", self.patience, "evaluations")
            return avg_reward, True  # True indicates early stopping should be triggered
            
        return avg_reward, False

    def save(self, filename="td3_model", directory="./checkpoints"):
        """
        Save the TD3 model and training history with timestamp.
        
        Args:
            filename: Name of the file to save (default: "td3_model")
            directory: Directory to save the model in (default: "./checkpoints")
        """
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_with_timestamp = f"{filename}_{timestamp}"
        
        print(f"\nSaving TD3 model to {directory}/{filename_with_timestamp}")
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename_with_timestamp))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename_with_timestamp))
        # Save training history
        history = {
            'reward_history': self.reward_history,
            'episode_lengths': self.episode_lengths,
            'total_episodes': self.total_episodes,
            'best_reward': self.best_reward,
            'patience_counter': self.patience_counter,
            'timestamp': timestamp,
            'filename': filename_with_timestamp
        }
        torch.save(history, '%s/%s_history.pth' % (directory, filename_with_timestamp))
        print("Model and training history saved successfully!")
        return filename_with_timestamp  # Return the filename with timestamp for reference

    def load(self, filename="td3_model", directory="./checkpoints"):
        """
        Load the TD3 model and training history.
        
        Args:
            filename: Name of the file to load (default: "td3_model")
            directory: Directory to load the model from (default: "./checkpoints")
            
        Returns:
            str: The full filename (with timestamp) of the loaded model
        """
        # If filename doesn't contain a timestamp, find the most recent model
        if '_' not in filename:
            # List all files in the directory
            files = os.listdir(directory)
            # Filter files that match the base filename
            matching_files = [f for f in files if f.startswith(filename) and f.endswith('_actor.pth')]
            if not matching_files:
                raise FileNotFoundError(f"No saved models found for {filename} in {directory}")
            # Sort by timestamp (newest first) and get the most recent
            latest_file = sorted(matching_files)[-1]
            # Extract the full filename without the '_actor.pth' suffix
            filename = latest_file[:-10]  # Remove '_actor.pth'
            print(f"\nFound most recent model: {filename}")
        
        print(f"\nLoading TD3 model from {directory}/{filename}")
        
        # Load actor and critic models
        actor_path = '%s/%s_actor.pth' % (directory, filename)
        critic_path = '%s/%s_critic.pth' % (directory, filename)
        history_path = '%s/%s_history.pth' % (directory, filename)
        
        if not os.path.exists(actor_path) or not os.path.exists(critic_path):
            raise FileNotFoundError(f"Model files not found in {directory}")
            
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        
        # Load training history if available
        try:
            if os.path.exists(history_path):
                history = torch.load(history_path)
                self.reward_history = history['reward_history']
                self.episode_lengths = history['episode_lengths']
                self.total_episodes = history['total_episodes']
                self.best_reward = history['best_reward']
                self.patience_counter = history['patience_counter']
                print(f"Training history loaded successfully!")
                print(f"Model was saved on: {history['timestamp']}")
                print(f"Best reward achieved: {self.best_reward:.2f}")
                print(f"Total episodes trained: {self.total_episodes}")
            else:
                print("No training history found, starting fresh")
        except Exception as e:
            print(f"Error loading training history: {str(e)}")
            print("Starting fresh with default values")
        
        print("Model loaded successfully!")
        return filename  # Return the full filename with timestamp 