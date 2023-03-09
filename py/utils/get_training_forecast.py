import tensorflow as tf
import numpy as np


def get_training_forecast(model_meta):
    loss_history = []
    for lr_meta in model_meta['lr_history']:
        for epoch in lr_meta['epoch_history']:
            loss_history.append(epoch['loss'])

    # Convert loss history to a numpy array
    loss_history = np.array(loss_history)

    # Create a small TensorFlow model with one hidden layer and train on the data
    model = tf.keras.Sequential([
        # tf.keras.layers.Dense(256, activation='sigmoid'),
        # tf.keras.layers.Dense(256, activation='sigmoid'),
        # tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dense(1, activation='linear')
        tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(1,)),
        tf.keras.layers.Dense(128, activation='sigmoid'),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    # Define a learning rate scheduler that reduces the learning rate by a factor of 10 every 5 epochs
    def lr_scheduler(epoch, lr):
        return lr / 1.01  # .1

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mean_squared_error'])

    xs = np.arange(len(loss_history)).reshape((-1, 1))
    ys = loss_history.reshape((-1, 1))

    # device = '/cpu:0'
    # with tf.device(device):
    print(xs, ys)

    early_callback = tf.keras.callbacks.EarlyStopping(
        monitor='mean_squared_error', patience=3, mode='min', verbose=1, restore_best_weights=True, min_delta=0, baseline=None)

    model.fit(xs, ys, epochs=75, callbacks=[tf.keras.callbacks.LearningRateScheduler(
        lr_scheduler), early_callback])

    # Make predictions with the trained model
    new_xs = np.arange(len(loss_history)*4).reshape((-1, 1))

    print(new_xs)

    new_ys = model.predict(new_xs).flatten()
    print(new_ys)

    # Return the new prediction array
    return new_ys
