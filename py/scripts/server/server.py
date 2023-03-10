from flask import Flask, jsonify, request
import numpy as np
from utils.load_model import load_model
from utils.get_all_model_names import get_all_model_names

app = Flask(__name__)

model_name = '../models/champ_he_M_v1/1.6669645971722076'
models_folder = '../models'

# Load the trained model
model = load_model(model_name)


@app.route('/predict/<model_name>', methods=['GET'])
def predict(model_name):

    print(model_name)

    # Get the input from the request parameters
    input_data = request.args.get('input')
    input_data = np.array(input_data.split(','), dtype=np.float32)

    # Reshape the input data
    input_data = input_data.reshape((1, 8, 8, 14))

    # Make a prediction with the model
    prediction = model.predict(input_data)

    # Convert the prediction to a string
    prediction_str = ','.join(str(x) for x in prediction[0])

    # Return the prediction as a JSON object
    return jsonify({'prediction': prediction_str})


@app.route('/models', methods=['GET'])
def example():
    model_names = get_all_model_names(models_folder)

    # Convert the string array to a JSON object
    return jsonify({'models': model_names})


if __name__ == '__main__':
    app.run(port=3600, debug=True)
