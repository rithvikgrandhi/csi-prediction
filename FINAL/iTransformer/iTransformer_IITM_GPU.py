# %%
import scipy.io
import numpy as np

# Load the .mat file
file_path = '/Users/rohitviswam/Desktop/IITM Mat file/EV_Rank_1_52_RBs_50_UEs_1000_snaps.mat'
data = scipy.io.loadmat(file_path)
# Extract the relevant data
data = data['EV_re_im_split']

feature_len=832


import numpy as np
import torch
from iTransformer import iTransformer
from iTransformer import iTransformer2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# %%
import numpy as np
import torch

def preprocess_CSI_data_step_by_step(data, lookback_len=10):
    """
    Preprocesses data to create input-target pairs with overlapping windows for CSI prediction.
    Args:
        data: numpy array, (5, 398, 256), the original dataset (5 UEs)
        lookback_len: int, number of past timesteps to consider as input
    
    Returns:
        inputs: torch Tensor, shape (samples, lookback_len, features)
        targets: torch Tensor, shape (samples, features)
    """
    num_UEs, num_timesteps, num_features = data.shape
    input_list = []
    target_list = []

    for ue in range(num_UEs):
        ue_data = data[ue]
        # Create input-target pairs with a step-by-step approach
        for i in range(0, num_timesteps - lookback_len):
            # Take the past `lookback_len` timesteps as input
            input_seq = ue_data[i: i + lookback_len]
            # Predict the next timestep following the input window
            target_seq = ue_data[i + lookback_len]

            input_list.append(input_seq)
            target_list.append(target_seq)

    # Convert lists to Torch tensors
    inputs = torch.tensor(input_list, dtype=torch.float32)
    targets = torch.tensor(target_list, dtype=torch.float32)

    return inputs, targets

# Example usage with dataset of shape (5, 398, 256)
lookback_len = 10  # Number of past timesteps to use as input

inputs, targets = preprocess_CSI_data_step_by_step(data, lookback_len)

print("Input shape:", inputs.shape)   # Expected shape: (samples, 10, 256)
print("Target shape:", targets.shape) # Expected shape: (samples, 256)


# %%
import torch
import torch.nn as nn
from iTransformer import iTransformer

# Set up parameters and data
num_UEs, num_timesteps, num_features = data.shape
lookback_len = 10
forecast_len = 1


# Define the iTransformer model
model = iTransformer(
    num_variates=num_features,
    lookback_len=lookback_len,
    dim=256,           # Model dimension
    depth=6,           # Number of layers
    heads=8,           # Number of attention heads
    dim_head=64,       # Dimension per head
    pred_length=forecast_len,  # Prediction horizon (1 in this case)
    num_tokens_per_variate=1,  # Single token per variate
    use_reversible_instance_norm=True
)



# %%
# Print the architecture by directly printing the model object
print(model)


# %%
# Define training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

inputs = inputs.to(device)
targets = targets.to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 200
batch_size = 64

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = int(len(inputs) / batch_size)
    
    for batch in range(num_batches):
     batch_inputs = inputs[batch * batch_size:(batch + 1) * batch_size]
     batch_targets = targets[batch * batch_size:(batch + 1) * batch_size]

     optimizer.zero_grad()
     preds = model(batch_inputs)[forecast_len].squeeze(1)  # Remove the singleton dimension
     loss = criterion(preds, batch_targets)
     loss.backward()
     optimizer.step()
     epoch_loss += loss.item()


    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / num_batches:.4f}")


# %%
import pandas as pd

def tabulate_predictions(inputs, targets, model, forecast_len=1, batch_size=64):
    """
    Tabulates actual vs. predicted results.
    Args:
        inputs: torch Tensor, input data used to predict
        targets: torch Tensor, actual target values
        model: trained PyTorch model for making predictions
        forecast_len: int, how many timesteps to forecast
        batch_size: int, batch size to use in prediction

    Returns:
        DataFrame containing actual and predicted results.
    """
    model.eval()
    all_preds = []
    all_targets = []

    num_batches = int(len(inputs) / batch_size)

    with torch.no_grad():
        for batch in range(num_batches):
            batch_inputs = inputs[batch * batch_size:(batch + 1) * batch_size]
            batch_targets = targets[batch * batch_size:(batch + 1) * batch_size]

            # Predict using the model
            preds = model(batch_inputs)[forecast_len].squeeze(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_targets.cpu().numpy())

    # Create DataFrame for easier tabulation and analysis
    df_results = pd.DataFrame({
        'Actual': [list(target) for target in all_targets],
        'Predicted': [list(pred) for pred in all_preds]
    })

    return df_results

# Example Usage:
results_df = tabulate_predictions(inputs, targets, model, forecast_len=1, batch_size=64)
print(results_df.head())  # Print the first few rows to verify


# %%
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(inputs, targets, model, forecast_len=1, num_samples=10, batch_size=64):
    """
    Plots a comparison between actual and predicted values.
    Args:
        inputs: torch Tensor, input data used to predict
        targets: torch Tensor, actual target values
        model: trained PyTorch model for making predictions
        forecast_len: int, how many timesteps to forecast
        num_samples: int, how many samples to plot
        batch_size: int, batch size to use in prediction

    Returns:
        None. Displays the plots.
    """
    model.eval()
    predictions = []
    actuals = []

    num_batches = min(int(len(inputs) / batch_size), num_samples)

    with torch.no_grad():
        for batch in range(num_batches):
            batch_inputs = inputs[batch * batch_size:(batch + 1) * batch_size]
            batch_targets = targets[batch * batch_size:(batch + 1) * batch_size]

            preds = model(batch_inputs)[forecast_len].squeeze(1)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(batch_targets.cpu().numpy())

    # Convert to numpy arrays for easy plotting
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Plot each sample
    plt.figure(figsize=(15, 8))
    for i in range(num_samples):
        plt.subplot(num_samples // 2, 2, i + 1)
        plt.plot(actuals[i], label='Actual')
        plt.plot(predictions[i], label='Predicted')
        plt.legend()
        plt.xlabel('Feature Index')
        plt.ylabel('Value')
        plt.title(f'Sample {i + 1}')

    plt.tight_layout()
    plt.show()

# Example usage:
plot_predictions(inputs, targets, model, forecast_len=1, num_samples=2, batch_size=64)


# %%
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from math import sqrt

def evaluate_model(model, inputs, targets, forecast_length_key=1):
    """
    Evaluates the model on given inputs and targets for a specific forecast horizon.

    Args:
        model: PyTorch model, the trained model for evaluation.
        inputs: torch.Tensor, inputs to the model (features).
        targets: torch.Tensor, actual target values.
        forecast_length_key: int, the specific forecast length to evaluate (e.g., 1 for 1-step prediction).
    
    Returns:
        mse: float, mean squared error.
        rmse: float, root mean squared error.
        mae: float, mean absolute error.
        r2: float, R-squared value.
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No need to track gradients
        predictions_dict = model(inputs)  # Obtain the dictionary output
        predictions = predictions_dict[forecast_length_key].squeeze()  # Extract the tensor for the desired forecast length
        targets = targets.squeeze()  # Ensure targets are squeezed too
        
        # Ensure predictions and targets are on the same device and have matching shapes
        mse = F.mse_loss(predictions, targets).item()
        rmse = sqrt(mse)
        mae = F.l1_loss(predictions, targets).item()
        r2 = r2_score(targets.cpu().numpy(), predictions.cpu().numpy())
    
    return mse, rmse, mae, r2

# Example usage:
# Ensure your `inputs` and `targets` are tensors on the correct device (cpu or cuda)
forecast_length = 1  # Example for a 1-step forecast; change to your desired value
mse, rmse, mae, r2 = evaluate_model(model, inputs, targets, forecast_length_key=forecast_length)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")


# %%
