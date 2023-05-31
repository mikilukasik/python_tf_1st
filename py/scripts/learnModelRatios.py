import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.callbacks import EarlyStopping


# Load the CSV file
data = pd.read_csv('moveScoreRatioStats-1684618303437.csv')

# Split the data into input (xs) and output (ys)
xs = data.iloc[:, -1].values
ys = data.iloc[:, :-1].values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.05)

# Define the TensorFlow model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='linear', input_shape=(1,)),
    # tf.keras.layers.Dropout(0.5),  # Add a dropout layer
    # tf.keras.layers.Dense(4096, activation=ELU()),
    tf.keras.layers.Dropout(0.5),  # Add a dropout layer
    tf.keras.layers.Dense(512, kernel_initializer='he_normal', activation='tanh'), tf.keras.layers.Dropout(0.5),  # Add a dropout layer
    tf.keras.layers.Dense(512, kernel_initializer='he_normal', activation='relu'), tf.keras.layers.Dropout(0.5),  # Add a dropout layer
    tf.keras.layers.Dense(512, kernel_initializer='he_normal', activation='sigmoid'), tf.keras.layers.Dropout(0.5),  # Add a dropout layer
    # tf.keras.layers.Dense(512, kernel_initializer='he_normal', activation=ELU()), tf.keras.layers.Dropout(0.5),  # Add a dropout layer
    # tf.keras.layers.Dense(512, kernel_initializer='he_normal', activation=ELU()), tf.keras.layers.Dropout(0.5),  # Add a dropout layer
    # tf.keras.layers.Dense(512, kernel_initializer='he_normal', activation=ELU()), tf.keras.layers.Dropout(0.5),  # Add a dropout layer
    # tf.keras.layers.Dense(512, kernel_initializer='he_normal', activation=ELU()), tf.keras.layers.Dropout(0.5),  # Add a dropout layer
    tf.keras.layers.Dense(512, kernel_initializer='he_normal', activation=ELU()),
    # tf.keras.layers.Dropout(0.5),  # Add a dropout layer
    # tf.keras.layers.Dense(64, activation=ELU()),
    # tf.keras.layers.Dropout(0.5),  # Add a dropout layer
    # tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dropout(0.5),  # Add a dropout layer
    tf.keras.layers.Dense(2, activation='linear')
])


# Compile the model
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(0.0001), loss='mean_absolute_error')

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='loss', patience=50, verbose=1)

# Train the model with early stopping
model.fit(x_train, y_train, epochs=5000, batch_size=16,  callbacks=[early_stopping])
# Evaluate the model
loss = model.evaluate(x_test, y_test)
print('Test loss:', loss)


# Generate input values from -300 to +300 with 0.1 increments
input_values = np.arange(-300, 300.1, 0.1)

# Predict the output values (ys) using the trained model
predicted_values = model.predict(input_values)




# Create a Scatter plot with hover information
fig = go.Figure()
fig.add_trace(go.Scatter(x=input_values, y=predicted_values[:, 0], mode='lines', name='Y1 Predicted'))
fig.add_trace(go.Scatter(x=input_values, y=predicted_values[:, 1], mode='lines', name='Y2 Predicted'))

fig.update_layout(
    title='Predicted Y Values',
    xaxis_title='X',
    yaxis_title='Y',
    hovermode='closest'
)

# Save the plot as an interactive HTML file
fig.write_html('predicted_values.html')


# # Create a PDF plot
# plt.figure(figsize=(10, 6))
# plt.plot(input_values, predicted_values[:, 0], label='Y1 Predicted')
# plt.plot(input_values, predicted_values[:, 1], label='Y2 Predicted')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Predicted Y Values')
# plt.legend()

# # Save the plot to a PDF file
# plt.savefig('predicted_values.pdf')
# plt.show()