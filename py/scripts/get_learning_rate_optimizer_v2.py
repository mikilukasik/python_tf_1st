import argparse
import os
import tensorflow as tf
import json
import numpy as np

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('metadata_folder', help='folder containing metadata.json')
args = parser.parse_args()

# Load metadata.json
metadata_path = os.path.join(args.metadata_folder, 'metadata.json')
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Extract training data from metadata
epochs = metadata['training_stats']['epochs']
training_data = []
for i in range(len(epochs)):
    xs = []
    if i < 10:
        num_missing_epochs = 10 - i
        xs += [0.0, 7.5] * num_missing_epochs

    for k in range(max(0, i-10), i):
        xs.append(epochs[k]['lr'])
        xs.append(epochs[k]['l'])

    # Calculate the percentage loss change and add it to the xs array
    loss_change = (epochs[i]['l'] - epochs[i-1]
                   ['l']) / epochs[i-1]['l']
    xs.append(loss_change)
    ys = epochs[i]['lr']
    training_data.append((xs, ys))

# Convert training data to NumPy arrays
X = [data[0] for data in training_data]
y = [data[1] for data in training_data]

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(200, activation='sigmoid', input_shape=(21,)),
    tf.keras.layers.Dense(400, activation='sigmoid'),
    tf.keras.layers.Dense(200, activation='sigmoid'),
    tf.keras.layers.Dense(1)
])

# Compile the model
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='mse')

# Log the model summary
model.summary()

# Define the learning rate scheduler
def lr_scheduler(epoch, lr):
    return lr * 0.98


# Define the early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=35, restore_best_weights=True
)

# Train the model
history = model.fit(X, y, epochs=750, verbose=1, callbacks=[early_stopping,
                    tf.keras.callbacks.LearningRateScheduler(lr_scheduler)])

# Log the history object
print(f"Training history: {history.history}")

# Save the model
model_path = os.path.join(args.metadata_folder, 'lr_estimator.h5')
model.save(model_path)

print(f"Model saved to {model_path}")
