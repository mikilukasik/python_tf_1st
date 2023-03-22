import tensorflow as tf
import numpy as np
from scipy.optimize import curve_fit
import math


# def smooth_data(data, window_size=10):
#     return data
#     weights = np.repeat(1.0, window_size)/window_size
#     smoothed_data = np.convolve(data, weights, 'valid')
#     return smoothed_data

def get_biased(in_array, num_segments=5):
    array = in_array[len(in_array)//5:]
    # return array
    segment_size = len(array) // num_segments
    concatenated_data = array.copy()

    for i in range(1, num_segments + 1):
        segment_start = len(array) - i * segment_size
        segment = array[segment_start:]
        concatenated_data = np.concatenate(
            (concatenated_data, segment), axis=0)
    return concatenated_data


def get_training_forecast_ai(model_meta):
    loss_history = []
    for epoch in model_meta['training_stats']['epochs']:
        loss_history.append(epoch['l'])

    # Convert loss history to a numpy array
    loss_history = np.array(loss_history)

    # Create a small TensorFlow model with one hidden layer and train on the data
    model = tf.keras.Sequential([
        # tf.keras.layers.Dense(256, activation='sigmoid', input_shape=(1,)),
        # tf.keras.layers.Dense(512, activation='sigmoid'),
        # tf.keras.layers.Dense(256, activation='sigmoid'),
        tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(1,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    # Define a learning rate scheduler that reduces the learning rate by a factor of 10 every 5 epochs
    def lr_scheduler(epoch, lr):
        return lr / 1.01  # .1

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mean_squared_error'])

    xs = get_biased(np.arange(len(loss_history))).reshape((-1, 1))
    ys = get_biased(loss_history).reshape((-1, 1))

    # pct_20 = math.ceil(len(xs) * 0.2)
    # pct_5 = math.ceil(len(xs) * 0.05)

    # xs = np.concatenate(
    #     (xs, xs[-pct_20:], xs[-pct_20:], xs[-pct_5:], xs[-pct_5:], xs[-pct_20:], xs[-pct_5:], xs[-pct_5:], xs[-pct_20:], xs[-pct_5:], xs[-pct_5:], xs[-pct_20:], xs[-pct_5:], xs[-pct_5:], xs[-pct_20:], xs[-pct_5:], xs[-pct_5:]), axis=0)
    # ys = np.concatenate(
    #     (ys, ys[-pct_20:], ys[-pct_20:], ys[-pct_5:], ys[-pct_5:], ys[-pct_20:], ys[-pct_5:], ys[-pct_5:], ys[-pct_20:], ys[-pct_5:], ys[-pct_5:], ys[-pct_20:], ys[-pct_5:], ys[-pct_5:], ys[-pct_20:], ys[-pct_5:], ys[-pct_5:]), axis=0)

    # device = '/cpu:0'
    # with tf.device(device):

    early_callback = tf.keras.callbacks.EarlyStopping(
        monitor='mean_squared_error', patience=3, mode='min', verbose=1, restore_best_weights=True, min_delta=0, baseline=None)

    model.fit(xs, ys, epochs=75, callbacks=[tf.keras.callbacks.LearningRateScheduler(
        lr_scheduler), early_callback])

    # Make predictions with the trained model
    new_xs = np.arange(math.ceil(len(loss_history)*3)).reshape((-1, 1))

    new_ys = model.predict(new_xs).flatten()

    input_data = np.array([len(loss_history)*3])
    # Reshape the input data to match the expected dimensions
    input_data = input_data.reshape((1, 1))
    prediction = model.predict(input_data)
    print(prediction)

    # Return the new prediction array
    return new_ys
