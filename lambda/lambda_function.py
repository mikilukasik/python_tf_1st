# import numpy as np
import os
import tensorflow as tf
from keras.models import model_from_json

model_folder = './champ'


def load_model(model_source: str):
    try:
        model_source = os.path.abspath(
            model_source)
        from_json = os.path.exists(os.path.join(model_source, "model.json"))
        if from_json:
            with open(os.path.join(model_source, "model.json"), "r") as f:
                model_json = f.read()

            model = model_from_json(model_json)
            try:
                model.load_weights(os.path.join(model_source, "weights.h5"))
                print('Weights loaded.', model_source)
            except Exception as e:
                print(f"Error loading weights: {str(e)}")
            print('JSON model loaded.', model_source)
        else:
            model = tf.keras.models.load_model(model_source)
            print('Keras model loaded.', model_source)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


model = load_model(model_folder)


def lambda_handler(event, context):

    return 'hello'

    # Get the input from the request parameters
    input_data = request.args.get('input')
    input_data = np.array(input_data.split(','), dtype=np.float32)

    # Reshape the input data
    input_data = input_data.reshape((1, 8, 8, 14))

    # Load the model from the cache or from disk

    # Make a prediction with the model
    prediction = model.predict(input_data)

    # Convert the prediction to a string
    prediction_str = ','.join(str(x) for x in prediction[0])

    # Return the prediction as a JSON object
    return jsonify({'prediction': prediction_str})
