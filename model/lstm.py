import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    A simple LSTM-based neural network model for sequence prediction tasks.

    Attributes:
        lstm (nn.LSTM): The LSTM layer.
        fc (nn.Linear): The fully connected layer that maps the LSTM output to the desired output size.
        hidden_size (int): The number of features in the hidden state of the LSTM.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the LSTMModel with the given input size, hidden size, and output size.

        Args:
            input_size (int): The number of expected features in the input.
            hidden_size (int): The number of features in the hidden state of the LSTM.
            output_size (int): The number of features in the output.
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x, hidden):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).
            hidden (tuple): A tuple containing the initial hidden state and cell state of the LSTM.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_size).
            tuple: The hidden state and cell state of the LSTM.
        """
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # Use the output of the last LSTM cell
        return out, hidden

    def init_hidden(self, device):
        """
        Initializes the hidden state and cell state of the LSTM to zeros.

        Args:
            device (torch.device): The device on which the tensors should be allocated.

        Returns:
            tuple: A tuple containing the initial hidden state and cell state of the LSTM.
        """
        return (torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device))