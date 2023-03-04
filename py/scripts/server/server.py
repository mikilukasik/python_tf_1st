from flask import Flask, jsonify, request
import numpy as np
from utils import load_model

app = Flask(__name__)

model_name = './models/save/champion/c4RESd2_S_v1/1.6521898729460578'

# Load the trained model
model = load_model(model_name)


@app.route('/predict', methods=['GET'])
def predict():
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


if __name__ == '__main__':
    app.run(port=3600, debug=True)
