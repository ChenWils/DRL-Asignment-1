# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import os
import torch
from simple_custom_taxi_env import SimpleTaxiEnv
import torch.optim as optim
import random
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
q_model = None  # Define q_model as global

def state_feature(obs):
    """
    Enhanced state features with:
    1. Normalized positions and distances
    2. Direction vectors to stations
    3. Passenger and destination status
    4. Obstacle information
    5. Relative position features
    """
    # Extract observation
    taxi_row, taxi_col, station0_row, station0_col, station1_row, station1_col,station2_row, station2_col, station3_row, station3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    
    # Calculate grid size for normalization
    grid_size = max(taxi_row, taxi_col, station0_row, station0_col, 
                    station1_row, station1_col, station2_row, station2_col,
                    station3_row, station3_col) + 1
    
    # Station positions
    stations = [(station0_row, station0_col), (station1_row, station1_col),
                (station2_row, station2_col), (station3_row, station3_col)]
    
    features = []
    
    # 1. Normalized taxi position
    features.extend([
        taxi_row / grid_size,
        taxi_col / grid_size
    ])
    
    # 2. Distance and direction to each station
    for station_row, station_col in stations:
        # Normalized Manhattan distance
        distance = (abs(taxi_row - station_row) + abs(taxi_col - station_col)) / (2 * grid_size)
        features.append(distance)
        
        # Direction vectors (normalized)
        features.extend([
            (station_row - taxi_row) / grid_size,
            (station_col - taxi_col) / grid_size
        ])
        
        # Binary indicators for being at station
        features.append(float(taxi_row == station_row and taxi_col == station_col))
    
    # 3. Obstacle information with directional context
    features.extend([
        obstacle_north,
        obstacle_south, 
        obstacle_east,
        obstacle_west,
        # Diagonal obstacles (combinations)
        obstacle_north and obstacle_east,
        obstacle_north and obstacle_west,
        obstacle_south and obstacle_east,
        obstacle_south and obstacle_west
    ])
    
    # 4. Passenger and destination status
    features.extend([
        passenger_look,
        destination_look,
        # Combined state for having both passenger and destination visible
        float(passenger_look == 1 and destination_look == 1)
    ])
    
    # 5. Closest station features
    min_distance = float('inf')
    closest_station_idx = 0
    for idx, (station_row, station_col) in enumerate(stations):
        dist = abs(taxi_row - station_row) + abs(taxi_col - station_col)
        if dist < min_distance:
            min_distance = dist
            closest_station_idx = idx
    
    # One-hot encoding of closest station
    for i in range(4):
        features.append(float(i == closest_station_idx))
    
    # Convert to tensor
    feature_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    return feature_tensor
    
    
def get_action(obs, eps=0.1):
    global q_model  # Declare q_model as global inside the function
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    q_table_path = "best_qtable.pt"
    
    if q_model is None and os.path.exists(q_table_path):
        states = state_feature(obs)
        input_dim = states.shape[0]  # Get correct input dimension

        q_model = DQN(input_dim, 6).to(device)  # Initialize model with correct input size
        q_model.load_state_dict(torch.load(q_table_path, map_location=device))
        q_model.eval()
        
    if q_model is None:
        print("NOT using q-table. RANDOM CHOICE!")
        # If model doesn't exist, return random actions
        action = random.choice([0, 1, 2, 3, 4, 5])
        return action
        
    # Epsilon-greedy policy for exploration during testing
    if random.random() < eps:
        action = random.choice([0, 1, 2, 3, 4, 5])
        return action
    
    # Preprocess state and get Q-values
    states = state_feature(obs)  # Make sure to define states
    with torch.no_grad():
        q_values = q_model(states)  # Use q_model instead of get_action.model
        action = torch.argmax(q_values).item()

    # Return action with highest Q-value
    return action
    # You can submit this random agent to evaluate the performance of a purely random strategy.
    # return random.choice([0, 1, 2, 3, 4, 5])
    

    
def reward_shaping(obs, next_obs, action, reward, position_history_length=5):
    """
    Shape rewards for:
    1. Finding and picking up passenger
    2. Delivering passenger
    3. General movement rewards/punishments
    4. Anti-oscillation penalties
    """
    # Extract current and next state information
    taxi_row, taxi_col, station0_row, station0_col, station1_row, station1_col, station2_row, station2_col, station3_row, station3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    
    next_taxi_row, next_taxi_col, _, _, _, _, _, _, _, _, \
    _, _, _, _, next_passenger_look, next_destination_look = next_obs
    
    # Initialize state tracking if not already done
    if not hasattr(reward_shaping, "pre_state"):
        reward_shaping.passenger_picked = False
        reward_shaping.position_history = []
        reward_shaping.action_history = []  # Track action history for oscillation detection
    
    shaped_reward = reward
    stations = [(station0_row, station0_col), (station1_row, station1_col), 
                (station2_row, station2_col), (station3_row, station3_col)]
    
    # 1. FINDING AND PICKING UP PASSENGER
    if not reward_shaping.passenger_picked:
        # Reward for discovering passenger            
        if passenger_look == 0 and next_passenger_look == 1:
            shaped_reward += 5.0
        
        # Reward for successful pickup
        if passenger_look == 1 and action == 4 and reward > 0:
            shaped_reward += 10.0
            reward_shaping.passenger_picked = True
        
        # Penalty for wrong pickup
        if passenger_look == 0 and action == 4:
            shaped_reward -= 2.0
    
    # 2. DELIVERING PASSENGER
    else:
        # Reward for discovering destination
        if destination_look == 0 and next_destination_look == 1:
            shaped_reward += 5.0
        
        # Reward for successful dropoff
        if destination_look == 1 and action == 5 and reward > 0:
            shaped_reward += 10.0
            reward_shaping.passenger_picked = False
        
        # Penalty for wrong dropoff
        if destination_look == 0 and action == 5:
            shaped_reward -= 2.0
            
    
    # 3. GENERAL MOVEMENT REWARDS/PUNISHMENTS
    
    # Penalty for not moving (hitting walls)
    if action < 4 and taxi_row == next_taxi_row and taxi_col == next_taxi_col:
        shaped_reward -= 0.2  # Reduced penalty
    
    # Update position history and penalize revisits
    reward_shaping.position_history.append((taxi_row, taxi_col))
    reward_shaping.action_history.append(action)  # Track actions for oscillation detection
    
    if len(reward_shaping.position_history) > position_history_length:
        reward_shaping.position_history.pop(0)
    
    if len(reward_shaping.action_history) > position_history_length:
        reward_shaping.action_history.pop(0)
    
    # 4. ANTI-OSCILLATION PENALTIES
    # Detect patterns like [0,2,0,2] (up-down-up-down) or [1,3,1,3] (right-left-right-left)
    if len(reward_shaping.action_history) >= 4:
        last_four = reward_shaping.action_history[-4:]
        
        # Check for up-down oscillation (actions 0 and 2)
        if (last_four[0] == 0 and last_four[1] == 2 and last_four[2] == 0 and last_four[3] == 2) or \
           (last_four[0] == 2 and last_four[1] == 0 and last_four[2] == 2 and last_four[3] == 0):
            shaped_reward -= 1.0  # Strong penalty for up-down oscillation
        
        # Check for left-right oscillation (actions 1 and 3)
        if (last_four[0] == 1 and last_four[1] == 3 and last_four[2] == 1 and last_four[3] == 3) or \
           (last_four[0] == 3 and last_four[1] == 1 and last_four[2] == 3 and last_four[3] == 1):
            shaped_reward -= 1.0  # Strong penalty for left-right oscillation
    
    # Also check for shorter oscillation patterns
    if len(reward_shaping.action_history) >= 2:
        last_two = reward_shaping.action_history[-2:]
        
        # Opposite direction pairs: (0,2), (2,0), (1,3), (3,1)
        opposite_pairs = [(0,2), (2,0), (1,3), (3,1)]
        
        if (last_two[0], last_two[1]) in opposite_pairs:
            shaped_reward -= 0.5  # Smaller penalty for immediate direction reversal
    
    # Position-based oscillation detection
    if len(reward_shaping.position_history) >= 6:
        # Check for position patterns like A-B-A-B-A-B (oscillating between two positions)
        last_six = reward_shaping.position_history[-6:]
        
        # Check if we're alternating between two positions
        if (last_six[0] == last_six[2] == last_six[4]) and (last_six[1] == last_six[3] == last_six[5]):
            shaped_reward -= 2.0  # Very strong penalty for position oscillation
        
        # Check for A-B-C-A-B-C pattern (three-position cycle)
        if (last_six[0] == last_six[3]) and (last_six[1] == last_six[4]) and (last_six[2] == last_six[5]):
            shaped_reward -= 1.5  # Strong penalty for three-position cycle
    
    # Check for position loops (returning to same position after a sequence of moves)
    if len(reward_shaping.position_history) >= 4:
        last_four = reward_shaping.position_history[-4:]
        if last_four[0] == last_four[-1] and len(set(last_four)) < 4:
            shaped_reward -= 1.0  # Penalty for short loops
    
    # Penalize based on number of revisits in history - with reduced penalty
    position_count = reward_shaping.position_history.count((taxi_row, taxi_col))
    if position_count > 2:  # Only penalize after more than 2 visits
        shaped_reward -= 0.1 * (position_count - 2)  # Reduced penalty
    
    # Small reward for getting closer to nearest station - with increased reward
    min_d = float('inf')
    for station in stations:
        distance = abs(taxi_row - station[0]) + abs(taxi_col - station[1])
        min_d = min(min_d, distance)
    
    if len(reward_shaping.position_history) > 1:
        prev_position = reward_shaping.position_history[-2]
        prev_min_d = float('inf')
        for station in stations:
            distance = abs(prev_position[0] - station[0]) + \
                      abs(prev_position[1] - station[1])
            prev_min_d = min(prev_min_d, distance)
        
        if min_d < prev_min_d:
            shaped_reward += 0.5  # Increased reward for progress
    
    return shaped_reward


class DQN(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super().__init__()
        self.first = True
        self.fc1 = nn.Sequential(
            nn.Linear(obs_dim, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.linear = nn.Linear(128, act_dim)
        
    def forward(self,obs):
        obs = self.fc1(obs)
        # if self.first:
            
        
        obs = self.fc2(obs)
        obs = self.linear(obs)
        return obs
    

class ReplayBuffer:
    def __init__(self, capacity=20000):
        """
        Initialize Replay Buffer
        Args:
            capacity: Maximum size of the buffer
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """
        Store transition in the buffer
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions
        Args:
            batch_size: Size of batch to sample
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to device
        return (torch.stack(states).to(device), 
                torch.tensor(actions, dtype=torch.long).to(device), 
                torch.tensor(rewards, dtype=torch.float32).to(device),
                torch.stack(next_states).to(device),
                torch.tensor(dones, dtype=torch.float32).to(device))
    
    def __len__(self):
        """Returns current size of the buffer"""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer = []
        self.position = 0
    
    
def train_agent():
    ### hyper param ###
    max_step = 200
    episodes = 20000
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9999
    batch_size = 128
    buffer_size = 100000
    min_samples = 1000  # Increased minimum samples before training
    lr = 0.0005  # Reduced learning rate for more stability
    lr_step_size = 2000  # Increased step size
    lr_gamma = 0.5  # Less aggressive learning rate decay
    best_reward = float('-inf')
    target_update = 30
    
    # Create models directory if it doesn't exist
    save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ###################
    
    ### env setting ###
    env = SimpleTaxiEnv()
    obs, _ = env.reset()
    state_tensor = state_feature(obs)
    input_dim = state_tensor.shape[0]
    
    # Initialize networks
    policy_net = DQN(input_dim, 6).to(device)
    target_net = DQN(input_dim, 6).to(device)  # Add target network
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr, weight_decay=1e-4)  # Added weight decay
    criterion = nn.SmoothL1Loss()  # Changed to Huber loss for stability
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    ####################
    
    rewards_per_episode = []
    recent_rewards = []  # Track recent rewards for early stopping
    
    for episode in tqdm(range(episodes)):
        obs, _ = env.reset()
        state_tensor = state_feature(obs)
        episode_reward = 0
        episode_step = 0
        done = False
        
        while not done and episode_step < max_step:
            episode_step += 1
            
            # Select action with epsilon-greedy
            if random.random() < epsilon:
                action = random.choice([0, 1, 2, 3, 4, 5])
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                action = torch.argmax(q_values).item()
            
            # Take action
            next_obs, reward, done, _ = env.step(action)
            next_state_tensor = state_feature(next_obs)
            shaped_reward = reward_shaping(obs, next_obs, action, reward)
            episode_reward += shaped_reward
            
            # Store transition in replay buffer
            replay_buffer.push(
                state_tensor,
                action,
                shaped_reward,
                next_state_tensor,
                done
            )
            
            # Train if enough samples
            if len(replay_buffer) > min_samples:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Get current Q values
                current_q_values = policy_net(states).gather(1, actions.unsqueeze(1))
                
                with torch.no_grad():
                    # Double DQN: Use policy_net to select actions, target_net to evaluate them
                    next_actions = policy_net(next_states).max(1)[1].unsqueeze(1)
                    next_q_values = target_net(next_states).gather(1, next_actions)
                    target_q_values = rewards.unsqueeze(1) + gamma * next_q_values * (1 - dones.unsqueeze(1))
                
                # Compute loss and optimize
                loss = criterion(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)  # Re-enabled gradient clipping
                optimizer.step()
            
            obs = next_obs
            state_tensor = next_state_tensor
            
        # Update target network periodically
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        scheduler.step()
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Print progress
        rewards_per_episode.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f},Best reward: {best_reward:.4f}")
            if avg_reward > best_reward:
                best_reward = avg_reward
                
                # Save both model and training state
                save_path = os.path.join(save_dir, f'dqn_checkpoint_ep{episode}_reward{best_reward:.2f}.pt')
                # Save checkpoint
                torch.save(policy_net.state_dict(), save_path)
                print(f"Saved checkpoint to {save_path} with reward: {best_reward:.2f}")

        
         # Save the model if the average reward is the best
        
    # return policy_net, rewards_per_episode


if __name__ == "__main__":
    train_agent()
    

