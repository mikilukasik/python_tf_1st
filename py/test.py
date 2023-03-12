import tensorflow as tf
import numpy as np


def extend_weights(old_weights, new_input_units):
    old_weights_shape = old_weights[0].shape
    old_input_units = old_weights_shape[0]
    new_weights_shape = (new_input_units, ) + old_weights_shape[1:]

    # Copy the existing weights to the new weight array
    new_weights = []
    for w in old_weights:
        new_w = np.zeros(new_weights_shape, dtype=w.dtype)
        new_w[:old_input_units, ...] = w
        new_weights.append(new_w)

    # Fill in the remaining weights by repeating the last few weights from the original array
    num_extra_units = new_input_units - old_input_units
    if num_extra_units > 0 and len(old_weights) > 0:
        extra_weights = [w[-1, ...] for w in old_weights]
        extra_weights = [np.tile(w, (num_extra_units // len(w) + 1,) +
                                 (1,) * (len(old_weights_shape) - 1)) for w in extra_weights]
        extra_weights = [w[:num_extra_units, ...] for w in extra_weights]
        new_weights = [np.concatenate([w, ew], axis=0)
                       for w, ew in zip(new_weights, extra_weights)]

    return new_weights


def copy_weights(old_layer, new_layer, new_input_units, weights_transformer=extend_weights):
    # Call the new layer with a random input to initialize its weights
    _ = new_layer(np.random.rand(10, new_input_units))

    # Get the weights of the old layer
    old_weights = old_layer.get_weights()

    # Transform the old weights for the new layer
    new_weights = weights_transformer(old_weights, new_input_units)

    # Set the weights of the new layer
    new_layer.set_weights(new_weights)


# Define an old layer
old_layer = tf.keras.layers.Dense(units=6, input_shape=(3,))
old_layer.build((None, 3))  # Build the layer with a dummy input shape

# Define a new layer with double the number of input units
new_layer = tf.keras.layers.Dense(units=6, input_shape=(8,))

# Copy the weights from the old layer to the new layer
copy_weights(old_layer, new_layer, new_input_units=8)

# Verify that the weights were copied correctly
old_weights = old_layer.get_weights()
new_weights = new_layer.get_weights()

assert np.array_equal(old_weights[0], new_weights[0][:6, ...])
assert np.array_equal(
    np.tile(old_weights[0][-1, ...], (2, 1)), new_weights[0][6:, ...])
assert np.array_equal(old_weights[1], new_weights[1])
