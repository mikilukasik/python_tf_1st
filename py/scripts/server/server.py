from flask import Flask, jsonify, request
import numpy as np
import threading
import os
from utils.load_model import load_model
from utils.get_all_model_names import get_all_model_names

app = Flask(__name__)

models_folder = '../models'
model_cache = {}
model_timers = {}


def clear_model_cache(model_name):
    print('Cleaning up after model:', model_name)
    model_cache.pop(model_name, None)
    model_timers.pop(model_name, None)


def replace_slash_and_dot(text):
    text = text.replace("-slash-", "/")
    text = text.replace("-dot-", ".")
    return text


@app.route('/predict/<model_name>', methods=['GET'])
def predict(model_name):

    print(model_name)

    # Get the input from the request parameters
    input_data = request.args.get('input')
    input_data = np.array(input_data.split(','), dtype=np.float32)

    # Reshape the input data
    input_data = input_data.reshape((1, 8, 8, 14))

    # Load the model from the cache or from disk
    model = model_cache.get(model_name)
    if model is None:
        model_path = os.path.join(
            models_folder, replace_slash_and_dot(model_name))
        print('Loading model:', model_name)
        model = load_model(model_path)
        model_cache[model_name] = model
        print('Loaded model:', model_name)

    # Make a prediction with the model
    prediction = model.predict(input_data)

    # Convert the prediction to a string
    prediction_str = ','.join(str(x) for x in prediction[0])

    # Set a timer to clear the model from the cache after a minute
    if model_name in model_timers:
        model_timers[model_name].cancel()
    timer = threading.Timer(60, clear_model_cache, args=[model_name])
    model_timers[model_name] = timer
    timer.start()

    # Return the prediction as a JSON object
    return jsonify({'prediction': prediction_str})


@app.route('/models', methods=['GET'])
def get_model_names():
    model_names = get_all_model_names(models_folder)

    # Convert the string array to a JSON object
    return jsonify({'models': model_names})


if __name__ == '__main__':
    app.run(port=3600, debug=True)
