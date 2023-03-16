import tensorflow as tf
import numpy as np
from scipy.optimize import curve_fit
import math


def smooth_data(data, window_size=10):
    weights = np.repeat(1.0, window_size)/window_size
    smoothed_data = np.convolve(data, weights, 'valid')
    return smoothed_data


def get_training_forecast(model_meta):
    loss_history = []
    for lr_meta in model_meta['lr_history']:
        for epoch in lr_meta['epoch_history']:
            loss_history.append(epoch['loss'])

    # Convert loss history to a numpy array
    loss_history = np.array(loss_history)

    # Define the function to fit
    def func(x, a, b, c, d):
        return d*(a/(x+b) + c)

    xs = smooth_data(np.arange(len(loss_history)))
    ys = smooth_data(loss_history)

    popt, pcov = curve_fit(func, xs, ys,  maxfev=10000)

    def forecaster(x):
        return popt[3] * (popt[0]/(x+popt[1]) + popt[2])

    new_xs = np.arange(len(loss_history)*2)
    new_ys = forecaster(new_xs)

    print(new_ys[-1])

    return new_ys


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

    xs = smooth_data(np.arange(len(loss_history))).reshape((-1, 1))
    ys = smooth_data(loss_history).reshape((-1, 1))

    pct_20 = math.ceil(len(xs)*0.2)
    pct_5 = math.ceil(len(xs)*0.05)
    print('pct_20', pct_20)
    print('pct_5', pct_5)

    xs = np.concatenate(
        (xs, xs[-pct_20:], xs[-pct_20:], xs[-pct_5:], xs[-pct_5:], xs[-pct_20:], xs[-pct_5:], xs[-pct_5:], xs[-pct_20:], xs[-pct_5:], xs[-pct_5:], xs[-pct_20:], xs[-pct_5:], xs[-pct_5:], xs[-pct_20:], xs[-pct_5:], xs[-pct_5:]), axis=0)
    ys = np.concatenate(
        (ys, ys[-pct_20:], ys[-pct_20:], ys[-pct_5:], ys[-pct_5:], ys[-pct_20:], ys[-pct_5:], ys[-pct_5:], ys[-pct_20:], ys[-pct_5:], ys[-pct_5:], ys[-pct_20:], ys[-pct_5:], ys[-pct_5:], ys[-pct_20:], ys[-pct_5:], ys[-pct_5:]), axis=0)

    # device = '/cpu:0'
    # with tf.device(device):

    early_callback = tf.keras.callbacks.EarlyStopping(
        monitor='mean_squared_error', patience=1, mode='min', verbose=1, restore_best_weights=True, min_delta=0, baseline=None)

    model.fit(xs, ys, epochs=75, callbacks=[tf.keras.callbacks.LearningRateScheduler(
        lr_scheduler), early_callback])

    # Make predictions with the trained model
    new_xs = np.arange(math.ceil(len(loss_history)*1.2)).reshape((-1, 1))

    new_ys = model.predict(new_xs).flatten()
    print(new_ys[-1])

    # Return the new prediction array
    return new_ys
