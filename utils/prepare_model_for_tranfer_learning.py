import tensorflow as tf
from typing import List


def prepare_model_for_transfer_learning(model: tf.keras.Model, new_layer_units: List[int]) -> tf.keras.Model:
    """
    Takes a Keras model and an array of integers specifying the number of units for the new dense layers to be added.
    Replaces the last set of dense layers in the model with new dense layers of similar structure, using the provided
    array of units. Returns the modified model with original weights and layers except the replaced dense layers.

    Parameters:
        model (keras.engine.training.Model): The Keras model to modify.
        new_layer_units (list of int): The array of integers specifying the number of units for the new dense layers.

    Returns:
        keras.engine.training.Model: The modified Keras model with new dense layers.

    Example:
        >>> base_model = keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        >>> new_layer_units = [1024, 512]
        >>> model = prepare_model_for_transfer_learning(base_model, new_layer_units)
    """

    # Find the index of the first non-dense layer from the end of the model
    last_dense_index = None
    for i, layer in reversed(list(enumerate(model.layers))):
        if isinstance(layer, tf.keras.layers.Dense):
            last_dense_index = i
        else:
            break

    if last_dense_index is None:
        raise ValueError("Model does not contain any dense layers")

    # Create new dense layers
    new_layers = []
    for units in new_layer_units:
        new_layers.append(tf.keras.layers.Dense(units, activation='relu'))

    # Create a new model with the same layers as the original model up to the last dense layer
    new_model_layers = model.layers[:last_dense_index]

    # Add the new dense layers and the output layer to the new model
    for layer in new_layers:
        new_model_layers.append(layer)
    new_model_layers.append(model.layers[-1])

    # Create a new model with the same weights and layers as the original model up to the last dense layer
    new_model = tf.keras.Model(
        inputs=model.input, outputs=new_model_layers[-1].output)
    new_model.build(input_shape=model.input_shape)

    return new_model
