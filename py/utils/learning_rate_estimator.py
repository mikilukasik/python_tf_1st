import numpy as np
import tensorflow as tf
import threading
import time


class ReinforcementLearningModel:
    def __init__(self, state_size=10, action_size=1, learning_rate=0.0001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            200, activation='relu', input_shape=(self.state_size,)))
        model.add(tf.keras.layers.Dense(400, activation='relu'))
        model.add(tf.keras.layers.Dense(200, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(
            lr=self.learning_rate))
        return model

    def predict(self, state):
        return self.model.predict(state)

    def train_on_batch(self, state, target):
        return self.model.train_on_batch(state, target)


class LearningRateEstimator:
    def __init__(self, model_meta, state_size=10, action_size=1, learning_rate=0.0001, discount_factor=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.model = ReinforcementLearningModel(
            state_size=self.state_size, action_size=self.action_size, learning_rate=self.learning_rate)
        self.loss_history = []
        self.lr_history = []

        # Initialize the loss and learning rate history from the metadata
        epochs = model_meta['training_stats']['epochs']
        for epoch in epochs:
            if 'l' in epoch and 'lr' in epoch:
                self.loss_history.append(epoch['l'])
                self.lr_history.append(epoch['lr'])

        self.training_thread = threading.Thread(
            target=self.train_reinforcement_learning_model)

    def after_epoch(self, loss, lr):
        self.loss_history.append(loss)
        self.lr_history.append(lr)

    def get_next_learning_rate(self):
        state = np.zeros((1, self.state_size))
        for i in range(self.state_size):
            if i < len(self.loss_history):
                state[0, i] = self.loss_history[-i - 1]
        prediction = self.model.predict(state)
        next_lr = prediction[0, 0]
        return next_lr

    def train_reinforcement_learning_model(self):
        interval = 0.001

        while True:
            time.sleep(interval)
            interval *= 1.0001

            num_samples = len(self.loss_history) - self.state_size
            # print('num_samples', num_samples)
            if num_samples > 0:
                X = np.zeros((num_samples, self.state_size))
                y = np.zeros((num_samples, self.action_size))

                for i in range(num_samples):
                    state = np.zeros((1, self.state_size))
                    for j in range(self.state_size):
                        state[0, j] = self.loss_history[-i - j - 1]
                    next_lr = self.lr_history[-i - self.state_size - 1]
                    action = np.array([next_lr])
                    X[i, :] = state
                    y[i, :] = action

                print(self.model.train_on_batch(X, y))

                # Decay the learning rate
                lr = tf.keras.backend.get_value(self.model.model.optimizer.lr)
                tf.keras.backend.set_value(
                    self.model.model.optimizer.lr, lr * self.discount_factor)

    def start_training_thread(self):
        self.training_thread.start()

    def stop_training_thread(self):
        self.training_thread.join()
