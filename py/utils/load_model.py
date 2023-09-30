from typing import Union, Dict
import os
import json
import tensorflow as tf
from keras.models import model_from_json
from .print_large import print_large


def load_model(model_source: str, quiet: bool = False) -> Union[tf.keras.Model, None]:
    """
    Load a Keras model from disk.

    Parameters:
        model_source (str): The path to the saved model directory.
        quiet (bool): If True, suppress all log messages.

    Returns:
        tf.keras.Model or None: The loaded Keras model, or None if an error occurred.
    """
    try:
        model_source = os.path.abspath(model_source)
        from_json = os.path.exists(os.path.join(model_source, "model.json"))
        if from_json:
            with open(os.path.join(model_source, "model.json"), "r") as f:
                model_json = f.read()

            model = model_from_json(model_json)
            try:
                model.load_weights(os.path.join(model_source, "weights.h5"))
                if not quiet:
                    print_large('Weights loaded.', model_source)
            except Exception as e:
                if not quiet:
                    print(f"Error loading weights: {str(e)}")
            if not quiet:
                print_large('JSON model loaded.', model_source)
        else:
            model = tf.keras.models.load_model(model_source)
            if not quiet:
                print_large('Keras model loaded.', model_source)
        return model
    except Exception as e:
        if not quiet:
            print(f"Error loading model: {str(e)}")
        return None


def load_model_meta(model_source: str, quiet: bool = False) -> Union[Dict, None]:
    """
    Load a metadata.json file from the given model directory.

    Parameters:
        model_source (str): The path to the saved model directory.
        quiet (bool): If True, suppress all log messages.

    Returns:
        dict or None: The loaded metadata as a dictionary, or None if an error occurred.
    """
    try:
        model_source = os.path.abspath(model_source)
        meta_path = os.path.join(model_source, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
                if not quiet:
                    print_large('Metadata loaded.', model_source)
            return metadata
        else:
            if not quiet:
                print(f"Metadata file not found in {model_source}")
            return {}
    except Exception as e:
        if not quiet:
            print(f"Error loading metadata: {str(e)}")
        return None
