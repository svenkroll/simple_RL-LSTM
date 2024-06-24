import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from agent.triple_action_agent import TripleActionAgent
from environment.simple_env import SimpleEnv


SEQUENCE_LENGTH = 3 # Sequence length determines how many data points our LSTM will process at a time

TRAIN_DATA_SIZE = 1000  # Number of data points to train on
TRAIN_EPISODES = 100  # Number of episodes to train
TRAIN_LEARNING_RATE = 0.001  # Learning rate for the optimizer

TEST_DATA_SIZE = 1000  # Number of data points to test on

LSTM_HIDDEN_SIZE = 6  # Number of hidden units in the LSTM
LSTM_INPUT_SIZE = 3  # Number of input units in the LSTM
LSTM_OUTPUT_SIZE = 3  # Number of output units in the LSTM

MODEL_SAVE_PATH = 'trained_model_test.pth'


def train():
    """
    Trains the TripleActionAgent using a simple environment and random data.

    This function initializes the environment and agent, then runs a training loop
    for a specified number of episodes. During each episode, the agent interacts
    with the environment, selects actions, and learns from the rewards received.
    The total rewards and losses per episode are recorded and plotted every 10 episodes.
    Finally, the trained model is saved to a file.

    Parameters:
    None

    Returns:
    None
    """
    device = torch.device("cpu")
    print("Using only the CPU for demonstration purposes")

    # Create an array of tuples, each containing three random values between 0 and 1
    data = np.array(
        [[random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)] for i in range(TRAIN_DATA_SIZE)])

    count_sum_greater_than_3 = np.sum(np    .sum(data[:-1] + data[1:], axis=1) > 3)
    print(f"The sum of a datapoint and the previous point which are greater than 3, {count_sum_greater_than_3} times.")

    env = SimpleEnv(data, device, SEQUENCE_LENGTH)

    agent = TripleActionAgent(input_size=LSTM_INPUT_SIZE, hidden_size=LSTM_HIDDEN_SIZE, output_size=LSTM_OUTPUT_SIZE, device=device, learning_rate=TRAIN_LEARNING_RATE)
    agent.model.train()  # Set the model to training mode

    num_episodes = TRAIN_EPISODES
    rewards_per_episode = []  # List to store total rewards per episode
    losses_per_episode = []  # List to store total losses per episode

    for episode in range(num_episodes):
        state = env.reset()
        hidden = agent.model.init_hidden(device)
        total_reward = torch.tensor(0.0, device=device)  # Initialize total reward as a tensor
        total_loss = torch.tensor(0.0, device=device)  # Initialize total loss as a tensor
        action_counts = {0: 0, 1: 0, 2: 0}  # Initialize action counts

        for t in range(len(env.data)):
            action, hidden = agent.select_action(state, hidden)
            next_state, reward, done = env.step(action)
            loss = agent.train(state, action, reward, next_state, done, hidden, hidden)
            state = next_state
            total_reward += reward  # Accumulate reward as a tensor
            total_loss += loss

            # Count actions get some understanding what the agent is doing
            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += 1

        rewards_per_episode.append(total_reward.item())  # Store total reward for the episode
        losses_per_episode.append(total_loss)  # Store total loss for the episode

        # Plot rewards and losses per episode
        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward.item()}, Total Loss: {total_loss.item()}, Action counts: {action_counts}")
            plt.figure()
            plt.plot(rewards_per_episode, label='Total Reward')
            plt.plot(losses_per_episode, label='Total Loss')
            plt.xlabel('Episode')
            plt.ylabel('Value')
            plt.title('Total Reward and Loss per Episode')
            plt.legend()
            plt.savefig('rewards_and_losses_per_episode.png')

    # Save the trained model
    torch.save(agent.model.state_dict(), MODEL_SAVE_PATH)


def test():
    device = torch.device("cpu")
    print("Using only the CPU for demonstration purposes")

    test_data = np.array([[random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)] for _ in range(TEST_DATA_SIZE)])
    env = SimpleEnv(test_data, device, SEQUENCE_LENGTH)
    agent = TripleActionAgent(input_size=LSTM_INPUT_SIZE, hidden_size=LSTM_HIDDEN_SIZE, output_size=LSTM_OUTPUT_SIZE, device=device)

    # Load the trained model
    agent.model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    agent.model.eval()  # Set the model to evaluation mode

    results = []

    state = env.reset()
    hidden = agent.model.init_hidden(device)
    episode_results = []

    for t in range(1, len(test_data)): # We test the model for eacht test datapoint
        action, hidden = agent.select_action(state, hidden)
        next_state, reward, done = env.step(action)
        episode_results.append((state.cpu().numpy(), action, reward.item()))
        state = next_state
        hidden = agent.model.init_hidden(device)
        if done:
            break

    results.append(episode_results)

    count_sum_greater_than_3 = 0

    # Count the number of times the sum of a datapoint and the previous point is greater than 3
    for step, (state, action, reward) in enumerate(episode_results):
        if (np.sum(state[2]) + np.sum(state[1])) > 3:
            count_sum_greater_than_3 += 10

    total_rewards = sum(reward for _, _, reward in episode_results)
    # Calculate the percentage of times the sum of a datapoint and the previous point is greater than 3
    percentage = total_rewards / (count_sum_greater_than_3 / 100) if total_rewards != 0 else 0
    print(f"Percentage: {percentage:.2f}%, Number of sums > 3: {count_sum_greater_than_3}, Total sum of rewards: {total_rewards}")


if __name__ == "__main__":
    train()
    test()