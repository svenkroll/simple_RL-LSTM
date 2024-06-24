import torch


class SimpleEnv:
    """
   A simple environment for reinforcement learning tasks.

   Attributes:
       data (torch.Tensor): The input data tensor.
       device (torch.device): The device (CPU or GPU) on which the tensor operations will be performed.
       current_step (int): The current step in the environment.
       seq_length (int): The length of the sequence to be considered for the state.
       state (torch.Tensor): The current state of the environment.
   """

    def __init__(self, data, device, seq_length=10):
        """
        Initializes the SimpleEnv with data, device, and sequence length.

        Args:
            data (list or numpy array): The input data to be converted to a tensor.
            device (torch.device): The device (CPU or GPU) on which the tensor operations will be performed.
            seq_length (int, optional): The length of the sequence to be considered for the state. Defaults to 10.
        """
        self.data = torch.tensor(data, device=device, dtype=torch.float32)
        self.device = device
        self.current_step = 0
        self.seq_length = seq_length
        self.state = self.get_state()

    def get_state(self):
        """
        Retrieves the current state of the environment.

        The state is a sequence of data points up to the current step, padded with zeros if necessary.

        Returns:
            torch.Tensor: The current state of the environment.
        """
        start = max(0, self.current_step - self.seq_length + 1)
        end = self.current_step + 1
        state = self.data[start:end]
        if len(state) < self.seq_length:
            padding = torch.zeros((self.seq_length - len(state), self.data.size(1)), device=self.device)
            state = torch.cat((padding, state), dim=0)
        return state

    def reset(self):
        """
        Resets the environment to the initial state.

        Returns:
            torch.Tensor: The initial state of the environment.
        """
        self.current_step = 0
        self.state = self.get_state()
        return self.state

    def step(self, action):
        """
        Takes a step in the environment based on the given action.

        Args:
            action (int): The action to be taken.

        Returns:
            tuple: A tuple containing the next state (torch.Tensor), the reward (torch.Tensor), and a boolean indicating if the episode is done.
        """
        reward = torch.tensor(0.0, device=self.device) # No reward in general
        if action == 0 and (torch.sum(self.state[-1]) + torch.sum(self.data[self.current_step - 1]) > 3):
            reward = torch.tensor(10.0, device=self.device)  # Higher reward when the sum is greater than 3

        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True
            self.state = torch.zeros_like(self.state)  # Dummy state when done
        else:
            self.state = self.get_state()
            done = False

        return self.state, reward, done
