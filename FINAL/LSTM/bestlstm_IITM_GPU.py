import scipy.io
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional, Input, Reshape
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print(f"TensorFlow Version: {tf.__version__}")

# Load the .mat file
file_path = '/Users/rohitviswam/Desktop/IITM Mat file/EV_Rank_1_52_RBs_50_UEs_1000_snaps.mat'
data = scipy.io.loadmat(file_path)

# Extract the relevant data
EV_data = data['EV_re_im_split']
data = EV_data
del EV_data
print(data.shape)


# Function to create sequences
def create_sequences(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(data.shape[0]):  # iterate over samples
        for j in range(data.shape[1] - n_steps_in - n_steps_out + 1):  # iterate over timesteps
            seq_x = data[i, j:j + n_steps_in]
            seq_y = data[i, j + n_steps_in:j + n_steps_in + n_steps_out]
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X), np.array(y)


timesteps_in = 10

X, y = create_sequences(data, timesteps_in,5)
print(f'X shape: {X.shape}, y shape: {y.shape}')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define a function to create the advanced LSTM model with gradient clipping
def create_model(dropout_rate=0.2, lstm_units=256, dense_units=512):
    model = Sequential()
    model.add(Input(shape=(timesteps_in, 832)))
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(lstm_units)))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(832*5))
    model.add(Reshape((5,832)))
    
    optimizer = Adam(clipvalue=1.0)

    model.compile(optimizer=optimizer, loss='mse')
    return model



best_params = {

    'dropout_rate': 0.2,
    'lstm_units': 4096,
    'dense_units': 4096,
    'batch_size': 64,
    'epochs': 250,
}

# Create and train the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

model = create_model(
    dropout_rate=best_params['dropout_rate'],
    lstm_units=best_params['lstm_units'],
    dense_units=best_params['dense_units'],
)





history = model.fit(
    X_train, y_train,
    batch_size=best_params['batch_size'],
    epochs=best_params['epochs'],
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1)


test_loss = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}")

predictions = model.predict(X_test)

# Flatten y_test and predictions along the first axis
y_test_flat = y_test.reshape(-1, y_test.shape[-1])
predictions_flat = predictions.reshape(-1, predictions.shape[-1])

# Calculate metrics
mse = mean_squared_error(y_test_flat, predictions_flat)
mae = mean_absolute_error(y_test_flat, predictions_flat)
r2 = r2_score(y_test_flat, predictions_flat)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R^2): {r2}')
