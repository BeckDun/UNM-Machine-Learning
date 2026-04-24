import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import time

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


def train_model(model, X_train, y_train, X_test, y_test, epochs=20, lr=0.001):
    mean_squared_error = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    train_losses = []
    test_losses = []

    training_start_time = time.time()
    for epoch in range(epochs):
        model.train()

        predictions = model(X_train)
        loss = mean_squared_error(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    training_end_time = time.time()

    training_time = training_end_time - training_start_time

    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = mean_squared_error(test_predictions, y_test)
        test_losses.append(test_loss.item())
    
    return {
        "test_losses": test_losses,
        "train_losses": train_losses,
        "training_time": training_time
    }



