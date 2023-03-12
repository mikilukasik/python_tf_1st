
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add, ReLU
from keras.initializers import he_normal
import numpy as np
import tensorflow as tf
import numpy as np

from utils.load_model import load_model
from utils.save_model import save_model


# def extend_weights(old_weights, new_input_units):
#     old_weights_shape = old_weights[0].shape
#     old_input_units = old_weights_shape[0]
#     new_weights_shape = (new_input_units, ) + old_weights_shape[1:]

#     # Copy the existing weights to the new weight array
#     new_weights = []
#     for w in old_weights:
#         new_w = np.zeros(new_weights_shape, dtype=w.dtype)
#         new_w[:old_input_units, ...] = w
#         new_weights.append(new_w)

#     # Fill in the remaining weights by repeating the last few weights from the original array
#     num_extra_units = new_input_units - old_input_units
#     if num_extra_units > 0:
#         extra_weights = [w[-num_extra_units:, ...] for w in old_weights]
#         extra_weights = [np.tile(w, (num_extra_units // w.shape[0] + 1,) + (
#             1,) * (len(old_weights_shape) - 1)) for w in extra_weights]
#         extra_weights = [w[:num_extra_units, ...] for w in extra_weights]
#         new_weights = [np.concatenate([w, ew], axis=0)
#                        for w, ew in zip(new_weights, extra_weights)]

#     return new_weights


# def copy_weights(old_layer, new_layer, new_input_units, weights_transformer=extend_weights):
#     # Call the new layer with a random input to initialize its weights
#     _ = new_layer(np.random.rand(10, new_input_units))

#     # Get the weights of the old layer
#     old_weights = old_layer.get_weights()

#     # Transform the old weights for the new layer
#     new_weights = weights_transformer(old_weights, new_input_units)

#     # Set the weights of the new layer
#     new_layer.set_weights(new_weights)


# def echo_weights(w):
#     return w

def duplicate_weights(w, desired_length):
    return [np.concatenate([w[0], w[0]], axis=1), np.concatenate([w[1], w[1]])]


def duplicate_weights2(w, desired_length=None):
    original_length = w[0].shape[1]
    if desired_length is None:
        desired_length = original_length * 2
    else:
        desired_length = max(original_length, min(
            desired_length, original_length * 2))

    new_weights = [
        np.zeros((w[0].shape[0], desired_length)), np.zeros(w[1].shape)]

    # Copy the old weights into the new array up to the original length
    new_weights[0][:, :original_length] = w[0]

    # Calculate the number of remaining weights to be filled from the end of the original array
    remaining_weights = desired_length - original_length

    # Copy the remaining weights from the end of the original array
    new_weights[0][:, original_length:] = w[0][:, -remaining_weights:]

    # Copy the bias term
    new_weights[1] = w[1]

    return new_weights


# def extend_array(arr, new_length):
#     # return arr

#     if new_length <= len(arr):
#         return arr

#     extension_length = new_length - len(arr)
#     extension_start = len(arr) - extension_length % len(arr)
#     extension = arr[extension_start:]
#     new_arr = arr + extension * (extension_length // len(arr))
#     new_arr += extension[:extension_length % len(arr)]

#     print(len(new_arr), len(arr))
#     return new_arr


def copy_weights(old_layer, new_layer, new_input_units, weight_extender=duplicate_weights):
    _ = new_layer(np.random.rand(10, new_input_units))
    old_weights = old_layer.get_weights()
    new_weights = weight_extender(old_weights, new_input_units)
    new_layer.set_weights(new_weights)


# Load the saved model
model_name = '../models/champ_he_M_v1/1.6645954327583314'
model = load_model(model_name)

# Set all layers except the new dense layers to not trainable
for layer in model.layers:
    layer.trainable = False

model.summary()

for i in range(len(model.layers)):
    print(i, model.layers[i], model.layers[i].trainable)

input = model.layers[0]

new_layer = Dense(1024, ELU(),  name="dense2")
copy_weights(model.layers[18], new_layer,
             new_input_units=1920)

x = new_layer(model.layers[17].output)

# output = model.layers[-1](Concatenate()([model.layers[13], x]))
output_layer = Dense(1837, 'softmax', name="output")
copy_weights(model.layers[-1], output_layer,
             new_input_units=1920, weight_extender=duplicate_weights2)


output = output_layer(Concatenate(
    name='concat_12')([model.layers[13].output, x]))


new_model = Model(inputs=input.output, outputs=output)

# Freeze all layers except the new dense layers
# for layer in new_model.layers[:-2]:
#     layer.trainable = False

# Print the updated model summary
new_model.summary()
for i in range(len(new_model.layers)):
    print(i, new_model.layers[i], new_model.layers[i].trainable)

save_model(new_model, '../models/champ_he_M_d1024-d1024_v1/_blank')

# # Set all layers except the new dense layers to not trainable
# for layer in model.layers:
#     layer.trainable = False

# # Set the new dense layers to trainable
# new_dense1.trainable = True
# new_dense2.trainable = True

# # # Set the output layer to not trainable
# # model.layers[-1].trainable = False

# # Replace the old dense layers with the new ones
# model.layers[16] = new_dense1
# model.layers[18] = new_dense2

# # Print the updated model summary
# model.summary()
