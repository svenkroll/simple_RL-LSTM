import random
import torch
import torch.optim as optim
import torch.nn as nn

from model.lstm import LSTMModel

GAMMA = 0.99 # Determines how much future rewards are worth compared to immediate rewards
INITIAL_EPSILON = 0.5  # Initial epsilon for ε-greedy
EPSILON_DECAY = 0.98  # Decay rate for epsilon
EPSILON_MIN = 0.01  # Minimum epsilon value


class TripleActionAgent:
    """
    A reinforcement learning agent that uses an LSTM model to select actions and learn from experiences.
    The agent follows an ε-greedy policy for action selection and uses Q-learning for training.
    """

    def __init__(self, input_size, hidden_size, output_size, device, learning_rate=0.001):
        """
        Initializes the TripleActionAgent with the given parameters.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden layer in the LSTM model.
            output_size (int): The number of possible actions.
            device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
            learning_rate (float, optional): The learning rate for the optimizer. Default is 0.001.
        """
        self.model = LSTMModel(input_size, hidden_size, output_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.gamma = GAMMA
        self.device = device
        self.epsilon = INITIAL_EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN

    def select_action(self, state, hidden):
        """
        Selects an action based on the current state and hidden state using an ε-greedy policy.

        Args:
            state (torch.Tensor): The current state of the environment.
            hidden (tuple): The hidden state of the LSTM model.

        Returns:
            tuple: A tuple containing the selected action (int) and the updated hidden state (tuple).
        """
        if self.model.training and random.random() < self.epsilon:
            return random.randint(0, 2), hidden  # Random action (exploration)
        else:
            with torch.no_grad():
                q_values, hidden = self.model(state.unsqueeze(0), hidden)  # Add batch dimension with 'unsqueeze(0)'!
            return q_values.max(1)[1].item(), hidden  # Best action predicted by the model (exploitation)

    def train(self, state, action, reward, next_state, done, hidden, next_hidden):
        """
        Trains the agent by updating the model's weights based on the given experience tuple.

        Args:
            state (torch.Tensor): The current state of the environment.
            action (int): The action taken by the agent.
            reward (float): The reward received after taking the action.
            next_state (torch.Tensor): The next state of the environment.
            done (bool): Whether the episode has ended.
            hidden (tuple): The hidden state of the LSTM model for the current state.
            next_hidden (tuple): The hidden state of the LSTM model for the next state.

        Returns:
            torch.Tensor: The loss value after the training step.
        """
        q_values, _ = self.model(state.unsqueeze(0), hidden)  # Add batch dimension with 'unsqueeze(0)'!
        next_q_values, _ = self.model(next_state.unsqueeze(0), next_hidden)  # Add batch dimension with 'unsqueeze(0)'!

        q_value = q_values[0, action]
        next_q_value = reward + self.gamma * next_q_values.max(1)[0] * (1 - done)

        # Compute the loss between the predicted Q-value and the target Q-value.
        # The loss function measures how well the model's predictions match the target values.
        # In this context, the target Q-value is the reward plus the discounted maximum future reward.
        # The loss is used to update the model's weights to improve its predictions.
        loss = self.loss_fn(q_value.unsqueeze(0).unsqueeze(0), next_q_value.unsqueeze(0))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon, which controls the exploration-exploitation trade-off.
        # The epsilon value is used to determine the probability of taking a random action.
        # If epsilon is high, the agent will explore the environment by taking random actions.
        # If epsilon is low, the agent will exploit the model's predictions by taking the best action.
        # The goal is to find a balance between exploration and exploitation that maximizes the agent's performance
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.detach()
