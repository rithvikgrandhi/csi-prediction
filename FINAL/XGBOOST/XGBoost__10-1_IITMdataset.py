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
# Parameters
n_samples = 50
n_timesteps = 832000
window_size = 10
test_size = 0.2  # 20% for testing

# Initialize lists to hold models and predictions/actuals for each UE
models = []
predictions = []
actuals = []

# %%
for ue in range(n_samples):
    # Extract data for this specific UE
    ue_data = data[ue, :]  # Shape: (n_timesteps,)
    
    # Create non-overlapping window pairs (input sequences and targets)
    X_ue = []
    y_ue = []
    
    for i in range(0, len(ue_data) - window_size, window_size):
        # Use the current window for input
        X_ue.append(ue_data[i:i + window_size])
        # Predict the next value immediately following the current window
        if i + window_size < len(ue_data):
            y_ue.append(ue_data[i + window_size])

    # Convert to NumPy arrays
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



# %%
# Assuming X_test and y_test are your test datasets
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE: ", rmse)

# %%
import numpy as np

# Assuming y contains your target variable values
min_value = np.min(y_ue)
max_value = np.max(y_ue)



# %%
accuracy = 100- (rmse /( max_value-min_value)) * 100

# %%
print("Accuracy is ", accuracy)

# %%
# Optionally save to a CSV file for further analysis
# final_results.to_csv("ue_predictions_comparison-10.csv", index=False)
