import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Clear previous TensorFlow session
import tensorflow.keras.backend as K
K.clear_session()

# Load your dataset (replace with actual data loading)
data = np.random.randn(50, 1000, 832)

# Create sequences
def create_sequences(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(data.shape[0]):
        for j in range(data.shape[1] - n_steps_in - n_steps_out + 1):
            X.append(data[i, j:j+n_steps_in, :])
            y.append(data[i, j+n_steps_in:j+n_steps_in+n_steps_out, :])
    return np.array(X), np.array(y)

n_steps_in = 5
n_steps_out = 1

X, y = create_sequences(data, n_steps_in, n_steps_out)
y = y.reshape((y.shape[0], y.shape[2]))  # Flatten the y array

# Ensure data is float32
X = X.astype('float32')
y = y.astype('float32')

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(LSTM(256, activation='relu', input_shape=(n_steps_in, X.shape[2])))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1]))  # Output layer

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

# Shuffle and batch the datasets
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(16).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

# Fit the model using the dataset API
history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, verbose=2, callbacks=[early_stopping])

# Evaluate the model
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Calculate R^2 Score
r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2}')

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f'Mean Absolute Percentage Error: {mape}%')

# Visualize Predictions
sample_index = 0  # Change this to visualize different samples
plt.figure(figsize=(10, 6))
plt.plot(y_test[sample_index], label='True')
plt.plot(y_pred[sample_index], label='Predicted')
plt.title('True vs Predicted')
plt.xlabel('Timestep')
plt.ylabel('Value')
plt.legend()
plt.show()
