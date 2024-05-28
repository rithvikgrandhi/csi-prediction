
import scipy.io
import numpy as np

# Load the .mat file
file_path = '/Users/rohitviswam/Desktop/IITM Mat file/EV_Rank_1_52_RBs_50_UEs_1000_snaps.mat'
data = scipy.io.loadmat(file_path)
# Extract the relevant data
data = data['EV_re_im_split']

feature_len=832

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


data_flattened = data.reshape(data.shape[0], -1)  # shape becomes (2100, 398*256)



data = data_flattened


data.shape


import numpy as np

# Assuming `data` is your loaded dataset with shape (2100, 101888)
num_ues, num_timesteps = data.shape
input_length = 10
target_length = 1  # predicting the next time step

# Prepare lists to hold input sequences and their targets
X = []
y = []

# Loop through each UE
for ue_idx in range(num_ues):
    for i in range(num_timesteps - input_length):
        # Extract the input sequence (current 10 time steps)
        input_seq = data[ue_idx, i:i + input_length]
        # Extract the target value (the next time step after the input sequence)
        target_value = data[ue_idx, i + input_length:i + input_length + target_length]
        
        X.append(input_seq)
        y.append(target_value)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Verify the shapes
print(f'Input shape: {X.shape}')  # Expected shape: (num_sequences, input_length)
print(f'Target shape: {y.shape}')  # Expected shape: (num_sequences, target_length)

# Example of reshaping for N-BEATS model if necessary
# (N-BEATS generally takes 3D input for each sequence, [batch_size, input_length, 1])
X = X[..., np.newaxis]
y = y[..., np.newaxis]

print(f'Input shape after reshaping: {X.shape}')  # (num_sequences, input_length, 1)
print(f'Target shape after reshaping: {y.shape}')  # (num_sequences, target_length, 1)



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Hyperparameters
input_length = 10
output_length = 1
batch_size = 64
epochs = 250
learning_rate = 0.001

# Creating a custom PyTorch dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

# Prepare the dataset
dataset = TimeSeriesDataset(X, y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the N-BEATS model
class NBEATSBlock(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_units):
        super(NBEATSBlock, self).__init__()
        layers = [nn.Linear(input_size, hidden_units), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        return self.output_layer(self.layers(x))

class NBEATS(nn.Module):
    def __init__(self, input_size, output_size, num_blocks, num_layers, hidden_units):
        super(NBEATS, self).__init__()
        self.blocks = nn.ModuleList([
            NBEATSBlock(input_size, output_size, num_layers, hidden_units) for _ in range(num_blocks)
        ])

    def forward(self, x):
        block_outputs = [block(x) for block in self.blocks]
        return sum(block_outputs)

# Initialize the model
model = NBEATS(input_length, output_length, num_blocks=3, num_layers=4, hidden_units=128)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_inputs, batch_targets in data_loader:
        optimizer.zero_grad()
        batch_inputs = batch_inputs.view(batch_inputs.size(0), -1)
        batch_targets = batch_targets.view(batch_targets.size(0), -1)

        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")



# Model is now trained; you can use it for inference or further evaluation.



import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Prepare evaluation data
def create_test_data(X, y):
    """Prepare a dataset without reshaping for evaluation."""
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Assuming X_test and y_test are properly split test data arrays
# If not, you can use slicing on X and y from the original dataset to create your test data
X_test, y_test = create_test_data(X[-500:], y[-500:])  # Example: take the last 500 sequences as test data

# Switch to evaluation mode
model.eval()

# Perform predictions
with torch.no_grad():
    X_test_flat = X_test.view(X_test.size(0), -1)
    predictions = model(X_test_flat)

# Convert tensors to numpy arrays for metric calculations
predictions_np = predictions.numpy().flatten()
y_test_np = y_test.numpy().flatten()

# Calculate evaluation metrics
mse = mean_squared_error(y_test_np, predictions_np)
mae = mean_absolute_error(y_test_np, predictions_np)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
