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
    
    
def get_action(obs, eps=0):
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

        q_model = DRQN(input_dim, 6).to(device)  # Initialize model with correct input size
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
    
class reward_state():
    def __init__(self):
        self.passenger_picked = False
        self.position_history = []
        self.action_history = []
        self.position_history_length = 5
    
def extra_reward(obs, next_obs, action, reward,reward_asisitant):
    """
    Shape rewards to complement original rewards:
    Original rewards:
    1. Task completion: +50
    2. Each step: -0.1
    3. Wrong pickup/dropoff: -10
    4. Hit wall: -5
    5. No fuel: -10
    """
    # Extract current and next state information
    taxi_row, taxi_col, station0_row, station0_col, station1_row, station1_col, station2_row, station2_col, station3_row, station3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    
    next_taxi_row, next_taxi_col, _, _, _, _, _, _, _, _, \
    _, _, _, _, next_passenger_look, next_destination_look = next_obs
    
    # Initialize state tracking if not already done
    
    shaped_reward = reward  # Start with original reward
    stations = [(station0_row, station0_col), (station1_row, station1_col), 
                (station2_row, station2_col), (station3_row, station3_col)]
    
    # 1. PROGRESS REWARDS
    if not reward_asisitant.passenger_picked:
        # Reward for discovering passenger (helps with exploration)           
        if passenger_look == 0 and next_passenger_look == 1:
            shaped_reward += 5.0
        
        # Additional reward for successful pickup (original already has -10 for wrong pickup)
        if passenger_look == 1 and action == 4 :  # Not a wall hit or wrong pickup
            shaped_reward += 10.0
            reward_asisitant.passenger_picked = True
        
        # Penalty for wrong pickup
        if passenger_look == 0 and action == 4:
            shaped_reward -= 3.0
    
    # 2. DELIVERING PASSENGER
    else:
        # Reward for discovering destination
        if destination_look == 0 and next_destination_look == 1:
            shaped_reward += 5.0
        
        # Additional reward for successful dropoff
        if destination_look == 1 and action == 5:  # Successful completion
            shaped_reward += 10.0
            reward_asisitant.passenger_picked = False
        
        # Penalty for wrong dropoff
        if destination_look == 0 and action == 5:
            shaped_reward -= 3.0
            
    
    # 3. GENERAL MOVEMENT REWARDS/PUNISHMENTS
    
    # Penalty for not moving (hitting walls)
    if  taxi_row == next_taxi_row and taxi_col == next_taxi_col:
        shaped_reward -= 1.0  # Reduced penalty
    
    # Update position history and penalize revisits
    reward_asisitant.position_history.append((taxi_row, taxi_col))
    reward_asisitant.action_history.append(action)
    
    if len(reward_asisitant.position_history) > reward_asisitant.position_history_length:
        reward_asisitant.position_history.pop(0)
    if len(reward_asisitant.action_history) > reward_asisitant.position_history_length:
        reward_asisitant.action_history.pop(0)
    
    # 3. ANTI-OSCILLATION PENALTIES (reduced magnitude since we have step penalty)
    # print(len(reward_asisitant.action_history))
    if len(reward_asisitant.action_history) >= 4:
        last_four = reward_asisitant.action_history[-4:]
        # print("Last four actions:", last_four)
        
        # Check for up-down oscillation
        if (last_four[0] == 0 and last_four[1] == 2 and last_four[2] == 0 and last_four[3] == 2) or \
           (last_four[0] == 2 and last_four[1] == 0 and last_four[2] == 2 and last_four[3] == 0):
            shaped_reward -= 3.0
            # print("-10")
        
        # Check for left-right oscillation
        if (last_four[0] == 1 and last_four[1] == 3 and last_four[2] == 1 and last_four[3] == 3) or \
           (last_four[0] == 3 and last_four[1] == 1 and last_four[2] == 3 and last_four[3] == 1):
            shaped_reward -= 3.0
            # print("-10")
    
    # Penalize immediate direction reversals
    if len(reward_asisitant.action_history) >= 2:
        last_two = reward_asisitant.action_history[-2:]
        
        # Opposite direction pairs: (0,2), (2,0), (1,3), (3,1)
        opposite_pairs = [(0,2), (2,0), (1,3), (3,1)]
        
        if (last_two[0], last_two[1]) in opposite_pairs:
            shaped_reward -= 3.0  # Smaller penalty for immediate direction reversal
            # print("-3")
    
    # 4. POSITION-BASED PENALTIES (reduced since we have step penalty)
    if len(reward_asisitant.position_history) >= 6:
        last_six = reward_asisitant.position_history[-6:]
        # Penalize two-position oscillation
        if (last_six[0] == last_six[2] == last_six[4]) and (last_six[1] == last_six[3] == last_six[5]):
            shaped_reward -= 1.5
        
        # Penalize three-position cycles
        if (last_six[0] == last_six[3]) and (last_six[1] == last_six[4]) and (last_six[2] == last_six[5]):
            shaped_reward -= 1.0
    
    # 5. PROGRESS TOWARD GOAL
    min_d = float('inf')
    target_stations = []
    
    # Determine relevant stations based on state
    if not reward_asisitant.passenger_picked and passenger_look == 1:
        target_stations = [(taxi_row, taxi_col)]  # Target current position for pickup
    elif reward_asisitant.passenger_picked and destination_look == 1:
        target_stations = [(taxi_row, taxi_col)]  # Target current position for dropoff
    else:
        target_stations = stations  # Search all stations
    
    # Calculate distance to nearest target
    for station in target_stations:
        distance = abs(taxi_row - station[0]) + abs(taxi_col - station[1])
        min_d = min(min_d, distance)
    
    # Reward progress toward target
    if len(reward_asisitant.position_history) > 1:
        prev_position = reward_asisitant.position_history[-2]
        prev_min_d = float('inf')
        for station in target_stations:
            distance = abs(prev_position[0] - station[0]) + abs(prev_position[1] - station[1])
            prev_min_d = min(prev_min_d, distance)
        
        if min_d < prev_min_d:
            shaped_reward += 3  # Small reward for getting closer to goal
    
    return shaped_reward


class DRQN(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=128, lstm_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        
        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Output layers
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize hidden state
        self.hidden = None
        
    def init_hidden(self, batch_size=1):
        """Initialize hidden state"""
        return (torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(device))
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0) if len(x.size()) > 1 else 1
        seq_length = x.size(1) if len(x.size()) > 2 else 1
        
        # Reshape input if it's a single sample
        if len(x.size()) == 1:
            x = x.unsqueeze(0).unsqueeze(1)  # Add batch and sequence dimensions
        elif len(x.size()) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            
        # Extract features
        features = self.feature_net(x.view(-1, x.size(-1)))
        features = features.view(batch_size, seq_length, -1)
        
        # LSTM forward pass
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        lstm_out, hidden = self.lstm(features, hidden)
        
        # Process all sequence outputs
        lstm_out = lstm_out.contiguous()
        
        # Dueling DQN architecture for each timestep
        advantage = self.advantage(lstm_out.view(-1, self.hidden_size))
        value = self.value(lstm_out.view(-1, self.hidden_size))
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        # Reshape back to (batch_size, seq_length, action_dim)
        q_values = q_values.view(batch_size, seq_length, -1)
        
        # Store hidden state
        self.hidden = hidden
        
        # If single sample, remove extra dimensions
        if batch_size == 1 and seq_length == 1:
            q_values = q_values.squeeze(0).squeeze(0)
            
        return q_values


class ReplayBuffer:
    def __init__(self, capacity=20000, sequence_length=8):
        """
        Initialize Replay Buffer
        Args:
            capacity: Maximum size of the buffer
            sequence_length: Length of sequences to sample
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.sequence_length = sequence_length
        self.episode_buffer = []  # Temporary buffer for current episode
        
    def push(self, state, action, reward, next_state, done,stop):
        """Store transition in the episode buffer"""
        self.episode_buffer.append((state, action, reward, next_state, done))
        
        if stop:
            # Store the entire episode
            if len(self.episode_buffer) >= self.sequence_length:
                if len(self.buffer) < self.capacity:
                    self.buffer.append(None)
                self.buffer[self.position] = list(self.episode_buffer)
                self.position = (self.position + 1) % self.capacity
            self.episode_buffer = []  # Clear episode buffer
        
    def sample(self, batch_size):
        """
        Sample a batch of sequences
        Args:
            batch_size: Size of batch to sample
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) where each is a sequence
        """
        episodes = random.sample(self.buffer[:-500], batch_size)
        
        # For each episode, randomly select a sequence of specified length
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        for episode in episodes:
            if len(episode) > self.sequence_length:
                start_idx = random.randint(0, len(episode) - self.sequence_length)
                sequence = episode[start_idx:start_idx + self.sequence_length]
            else:
                sequence = episode[-self.sequence_length:]
                
            # Unzip the sequence
            states, actions, rewards, next_states, dones = zip(*sequence)
            
            batch_states.append(torch.stack(states))
            batch_actions.append(torch.tensor(actions, dtype=torch.long))
            batch_rewards.append(torch.tensor(rewards, dtype=torch.float32))
            batch_next_states.append(torch.stack(next_states))
            batch_dones.append(torch.tensor(dones, dtype=torch.float32))
        
        # Stack all sequences
        return (torch.stack(batch_states).to(device),
                torch.stack(batch_actions).to(device),
                torch.stack(batch_rewards).to(device),
                torch.stack(batch_next_states).to(device),
                torch.stack(batch_dones).to(device))
    
    def __len__(self):
        """Returns current size of the buffer"""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer = []
        self.position = 0
        self.episode_buffer = []
    
    
def train_agent():
    ### hyper param ###
    max_step = 320
    episodes = 20000
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05  # Increased minimum epsilon
    epsilon_decay = 0.9999  # Faster epsilon decay
    batch_size = 128  # Smaller batch size for sequences
    sequence_length = 8  # Length of sequences to train on
    buffer_size = 10000  # Reduced buffer size since we store episodes
    min_samples = 1000  # Minimum episodes before training
    lr = 0.0001  # Reduced learning rate for stability
    lr_step_size = 1000
    lr_gamma = 0.95
    best_reward = float('-inf')
    target_update = 100  # Less frequent target updates
    hidden_size = 128  # LSTM hidden size
    lstm_layers = 1  # Number of LSTM layers
    
    # Create models directory if it doesn't exist
    save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ###################
    
    ### env setting ###
    # env = SimpleTaxiEnv()
    from complicated_env import ComplexTaxiEnv
    env = ComplexTaxiEnv()
    
    obs, _ = env.reset()
    state_tensor = state_feature(obs)
    input_dim = state_tensor.shape[0]
    
    # Initialize networks
    policy_net = DRQN(input_dim, 6, hidden_size, lstm_layers).to(device)
    target_net = DRQN(input_dim, 6, hidden_size, lstm_layers).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()
    replay_buffer = ReplayBuffer(capacity=buffer_size, sequence_length=sequence_length)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    
    rewards_per_episode = []
    
    for episode in tqdm(range(episodes)):
        obs, _ = env.reset()
        state_tensor = state_feature(obs)
        episode_reward = 0
        episode_step = 0
        done = False
        reward_asistant = reward_state()
        
        # Reset hidden states at the start of each episode
        policy_net.hidden = policy_net.init_hidden()
        target_net.hidden = target_net.init_hidden()
        
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
            shaped_reward = extra_reward(obs, next_obs, action, reward,reward_asistant)
            episode_reward += shaped_reward
            
            stop = done or episode_step > max_step
            # Store transition in replay buffer
            replay_buffer.push(
                state_tensor,
                action,
                shaped_reward,
                next_state_tensor,
                done,
                stop
            )
            
            # Train if enough samples
            if len(replay_buffer) > min_samples:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Reset hidden states for both networks
                policy_net.hidden = policy_net.init_hidden(batch_size)
                target_net.hidden = target_net.init_hidden(batch_size)
                
                # Get current Q values for entire sequences
                current_q_values = policy_net(states)  # Shape: [batch_size, seq_length, n_actions]
                actions = actions.unsqueeze(-1)  # Shape: [batch_size, seq_length, 1]
                current_q_values = current_q_values.gather(2, actions)  # Shape: [batch_size, seq_length, 1]
                
                with torch.no_grad():
                    # Get next Q values for entire sequences
                    next_q_values = target_net(next_states)  # Shape: [batch_size, seq_length, n_actions]
                    next_actions = policy_net(next_states).argmax(dim=2, keepdim=True)  # Shape: [batch_size, seq_length, 1]
                    next_q_values = next_q_values.gather(2, next_actions)  # Shape: [batch_size, seq_length, 1]
                    next_q_values = next_q_values.squeeze(-1)  # Shape: [batch_size, seq_length]
                    
                    # Calculate target Q values
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)
                
                # Reshape current Q values to match target shape
                current_q_values = current_q_values.squeeze(-1)  # Shape: [batch_size, seq_length]
                
                # Compute loss and optimize
                loss = criterion(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()
                
            
            obs = next_obs
            state_tensor = next_state_tensor
            
        if episode + 1 > min_samples:    
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
        # Update target network periodically
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        scheduler.step()
        
        
        # Print progress
        rewards_per_episode.append(episode_reward)
        
        if (episode + 1) % 100 == 0 and episode + 1 > min_samples:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}, Best reward: {best_reward:.4f}")
            if avg_reward > best_reward:
                best_reward = avg_reward
                save_path = os.path.join(save_dir, f'drqn_checkpoint_ep{episode}_reward{best_reward:.2f}.pt')
                torch.save(policy_net.state_dict(), save_path)
                print(f"Saved checkpoint to {save_path} with reward: {best_reward:.2f}")


if __name__ == "__main__":
    train_agent()
    

