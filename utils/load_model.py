from typing import Union
import os
import tensorflow as tf
from keras.models import model_from_json
from utils import print_large


def load_model(model_source: str) -> Union[tf.keras.Model, None]:
    """
    Load a Keras model from disk.

    Parameters:
        model_source (str): The path to the saved model directory.

    Returns:
        tf.keras.Model or None: The loaded Keras model, or None if an error occurred.
    """

    try:
        from_json = os.path.exists(os.path.join(model_source, "model.json"))
        if from_json:
            with open(os.path.join(model_source, "model.json"), "r") as f:
                model_json = f.read()

            model = model_from_json(model_json)
            model.load_weights(os.path.join(model_source, "weights.h5"))
            print_large('JSON model loaded.', model_source)
        else:
            model = tf.keras.models.load_model(model_source)
            print_large('Keras model loaded.', model_source)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None
