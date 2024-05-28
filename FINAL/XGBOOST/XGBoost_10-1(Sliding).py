# %%
import torch
import pandas as pd
import numpy as np
import tensorflow as tf

# %%
import scipy.io
import numpy as np

# Load the .mat file
file_path = '/Users/rohitviswam/Desktop/IITM Mat file/EV_Rank_1_52_RBs_50_UEs_1000_snaps.mat'
data = scipy.io.loadmat(file_path)

# Inspect the structure of the loaded data
data.keys()


# %%
# Extract the relevant data
EV_data = data['EV_re_im_split']

# %%
data = EV_data

# %%
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# %%
data_flattened = data.reshape(data.shape[0], -1)  # shape becomes (2100, 398*256)


# %%
data = data_flattened

# %%
data.shape

# %%
# Parameters
n_samples = 50
n_timesteps = 832000
window_size = 10
test_size = 0.2  # 20% for testing

# Initialize a list to hold models for each UE
models = []

# Initialize lists to hold performance evaluation data
predictions = []
actuals = []

# %%
for ue in range(n_samples):
    # Extract data for this specific UE
    ue_data = data[ue, :]  # Shape: (n_timesteps,)
    
    # Create rolling window pairs (input sequences and targets)
    X_ue = []
    y_ue = []
    
    for i in range(len(ue_data) - window_size):
        X_ue.append(ue_data[i:i + window_size])
        y_ue.append(ue_data[i + window_size])
    
    X_ue = np.array(X_ue)
    y_ue = np.array(y_ue)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_ue, y_ue, test_size=test_size, random_state=42)
    
    # Initialize the XGBoost model
    model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, objective='reg:squarederror')
    
    # Train the model on the training set
    model.fit(X_train, y_train)
    
    # Make predictions on the testing set
    y_pred = model.predict(X_test)
    
    # Save the predictions and actuals for evaluation
    predictions.append(y_pred)
    actuals.append(y_test)
    
    # Save the model for this UE
    models.append(model)





