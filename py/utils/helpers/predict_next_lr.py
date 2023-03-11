import os
import tensorflow as tf
import numpy as np

# Load the saved model
model_path = os.path.join('./utils/helpers/lr_estimator.h5')
model = tf.keras.models.load_model(model_path)


def predict_next_lr(model_meta):
    epochs = model_meta['training_stats']['epochs']

    loss_history = []
    lr_history = []

    for i in range(len(epochs) - 10, len(epochs)):
        if i < 0:
            lr_history.append(0)
            loss_history.append(7.5)
        else:
            lr_history.append(epochs[i]['lr'])
            loss_history.append(epochs[i]['l'])

    # Convert the loss and learning rate histories to a numpy array
    loss_history = np.array(loss_history)
    lr_history = np.array(lr_history)

    # Create the input tensor for the model
    xs = []
    for i in range(10):
        xs.append(lr_history[i])
        xs.append(loss_history[i])
    xs.append(1)

    # Reshape the input tensor to match the model's input shape
    xs = np.array(xs).reshape(1, -1)

    # Use the model to predict the next learning rate
    y_pred = model.predict(xs)[0][0]

    return y_pred
