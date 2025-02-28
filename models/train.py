import torch
import torch.nn.functional as F
from models.inference import ConnectFourModel
from environment.state import GameState, Winner
import numpy as np
from typing import List, Tuple
from models.utils import fill_board

def train(policy_model: ConnectFourModel, batch_size=1, num_epochs=10, discount_factor=0.95):
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        games_played = 0
        wins = 0
                    
        # 1. Generate trajectories through self-play
        trajectories = generate_self_play_trajectories(policy_model, batch_size)
        
        # 2. Calculate returns and policy loss
        total_loss = 0
        for trajectory in trajectories:
            states_trajectory, actions_trajectory, rewards_trajectory = trajectory
            
            # Track win rate
            if rewards_trajectory[-1] > 0:
                wins += 1
            games_played += 1
            
            states_tensor = torch.stack(states_trajectory, dim=0)
            actions_tensor = torch.tensor(actions_trajectory, dtype=torch.long)
         
            # Calculate discounted returns
            returns = calculate_discounted_returns(rewards_trajectory, discount_factor)
            returns_tensor = torch.tensor(returns, dtype=torch.float32)
            
            # Get action probabilities from the model
            
            # Reshape from (T, B, C, height, width) to (T*B, C, height, width)
            T, B, C, H, W = states_tensor.shape
            reshaped_states = states_tensor.view(-1, C, H, W)
            
            # Forward pass
            logits = policy_model(reshaped_states)
            
            # Reshape logits back to match the original batch structure
            logits = logits.view(T, B, -1)
            
            # Convert logits to probabilities using softmax
            log_probs = F.log_softmax(logits, dim=-1)
            # Select log probabilities for the actions that were taken
            selected_log_probs = log_probs[torch.arange(len(actions_trajectory)), :, actions_tensor]
            
            # Calculate policy gradient loss
            loss = -torch.mean(selected_log_probs * returns_tensor)
            total_loss += loss
        
        # Average loss over all trajectories in the batch
        avg_loss = total_loss / len(trajectories)
        
        # Update the model
        optimizer.zero_grad()
        avg_loss.backward() # type: ignore
        optimizer.step()
        
        epoch_loss = avg_loss.item() # type: ignore
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Win Rate: {wins/games_played:.4f}")
    
    return policy_model

def generate_self_play_trajectories(policy_model: ConnectFourModel, num_games: int) -> List[Tuple[List[torch.Tensor], List[int], List[float]]]:
    """
    Generate trajectories by having the policy play against itself.
    
    Returns:
        List of trajectories, where each trajectory is a tuple of 
        (states, actions, rewards) for one complete game.
    """
    trajectories = []
    
    for _ in range(num_games):
        game = GameState()
        states = []
        actions = []
        
        # Play until game is over
        while not game.is_game_over():
            # Convert current board state to tensor

            board_tensor = fill_board(game, game.current_player).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            states.append(board_tensor)
            
            # Get action probabilities from the model
            with torch.no_grad():
                logits = policy_model(board_tensor)
                # Use softmax to get probabilities
                probs = F.softmax(logits, dim=-1)
                
            # Sample an action from the probability distribution
            action_probs = probs.squeeze(0).numpy()
            
            action = np.random.choice(7, p=action_probs)
            actions.append(action)
            
            # Take the action
            game.add(action)
        
        # Calculate rewards
        winner = game.get_winner()
        rewards = [0] * len(actions)
        
        if winner == Winner.RED:
            # Reward actions that led to RED winning
            # Since players alternate, even indices are RED's moves
            rewards = [1.0 if i % 2 == 0 else -1.0 for i in range(len(actions))]
        elif winner == Winner.YELLOW:
            # Reward actions that led to YELLOW winning
            # Odd indices are YELLOW's moves
            rewards = [-1.0 if i % 2 == 0 else 1.0 for i in range(len(actions))]
        # Draw results in zero rewards
        
        trajectories.append((states, actions, rewards))
    
    return trajectories

def calculate_discounted_returns(rewards: List[float], discount_factor: float) -> List[float]:
    """
    Calculate discounted returns for each step in an episode.
    """
    returns = []
    G = 0
    
    # Calculate returns from the end of the episode
    for r in reversed(rewards):
        G = r + discount_factor * G
        returns.insert(0, G)
        
    return returns

if __name__ == "__main__":
    model = ConnectFourModel()
    model = train(model, batch_size=64, num_epochs=512, discount_factor=0.95)
