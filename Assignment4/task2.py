import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class RNN(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 64, output_size = 1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first = True)
        self.fully_connected_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, hidden_state = self.rnn(x)
        output = output[:, -1, :]
        output = self.fully_connected_layer(output)
        return output
