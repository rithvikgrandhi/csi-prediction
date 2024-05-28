import scipy.io
import numpy as np

# Load the .mat file
file_path = '../../EV_Rank_1_52_RBs_50_UEs_1000_snaps.mat'
data = scipy.io.loadmat(file_path)
# Extract the relevant data
data = data['EV_re_im_split']

feature_len=832

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


data = data.reshape(data.shape[0], -1)


import numpy as np
# Set up parameters
num_UEs, num_timesteps = data.shape
sequence_length = 10  # Use the first 10 values as input
stride = 1  # Moving one step at a time

X, y = [], []

# Loop through each UE
for ue_idx in range(num_UEs):
    ue_data = data[ue_idx]
    num_sequences = (num_timesteps - sequence_length) // stride

    input_indices = np.arange(sequence_length) + np.arange(num_sequences)[:, None]
    input_seq = ue_data[input_indices]

    output_indices = np.arange(sequence_length, num_sequences + sequence_length)
    output_value = ue_data[output_indices]

    X.append(input_seq)
    y.append(output_value)

X = np.concatenate(X)
y = np.concatenate(y)

print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")


# Define the train-test split ratio (e.g., 80% train, 20% test)
train_ratio = 0.8
train_size = int(len(X) * train_ratio)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing data shapes: X_test: {X_test.shape}, y_test: {y_test.shape}")


import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, Dropout, Add, Dense, Flatten
from tensorflow.keras.optimizers import Adam

def residual_block(x, dilation_rate, num_filters, kernel_size, dropout_rate=0.0):
    conv1 = Conv1D(num_filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
    bn1 = BatchNormalization()(conv1)
    act1 = ReLU()(bn1)
    drop1 = Dropout(dropout_rate)(act1)


    if x.shape[-1] != num_filters:
        x = Conv1D(num_filters, kernel_size=1)(x)

    out = Add()([x, drop1])
    out = ReLU()(out)
    return out

def build_tcn_model(input_shape, num_blocks=2, num_filters=32, kernel_size=3, dropout_rate=0.2):
    inputs = Input(shape=input_shape)
    x = inputs

    for i in range(num_blocks):
        dilation_rate = 2 ** i
        x = residual_block(x, dilation_rate, num_filters, kernel_size, dropout_rate)

    x = Flatten()(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

input_length = 10  # The number of previous time steps used for input
num_features = 1  # Modify according to your dataset


import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


num_features = 1  # Adjust based on the dataset (e.g., 1 for univariate, multiple if needed)
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], num_features))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], num_features))

model = build_tcn_model(input_shape=(X_train.shape[1], num_features))

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_tcn_model.keras', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    X_train_reshaped, y_train,
    validation_data=(X_test_reshaped, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)
