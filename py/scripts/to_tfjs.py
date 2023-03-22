import os
import sys
from datetime import datetime

from utils.find_most_recent_model import find_most_recent_model
from utils.load_model import load_model_meta
from utils.plot_model_meta import plot_model_meta

import json
import tensorflow as tf
import tensorflowjs as tfjs


def convert_and_save_model(model_folder, output_dir, shard_size_bytes=1024 * 1024 * 4):
    model_path = os.path.join(model_folder, 'model.h5')
    model = tf.keras.models.load_model(model_path)
    tfjs.converters.save_keras_model(
        model, output_dir, shard_size_bytes=shard_size_bytes)


def save_model_meta_as_json(model_meta, output_path):
    with open(output_path, 'w') as f:
        json.dump(model_meta, f, indent=4)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_metadata.py folder [forecast]")
        sys.exit(1)

    folder = sys.argv[1]

    if not os.path.isdir(folder):
        print(folder, "is not a valid directory")
        sys.exit(1)

    model_folder = find_most_recent_model(folder)

    if not model_folder:
        print("No model.json files found in", folder)
        exit()

    model_meta = load_model_meta(model_folder)

    output_dir = os.path.join(model_folder, 'tfjs_model')
    os.makedirs(output_dir, exist_ok=True)

    # Save the model as a TensorFlow.js model
    convert_and_save_model(model_folder, output_dir)

    # Save the model metadata as a JSON file
    model_meta_json_path = os.path.join(output_dir, 'model_meta.json')
    save_model_meta_as_json(model_meta, model_meta_json_path)
