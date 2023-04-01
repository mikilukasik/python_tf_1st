import keras
from keras.models import model_from_json

# Load the model structure from the JSON file
with open("models/plain_16_v5/1.654546069463094/model.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Load the model weights from the .h5 file
model.load_weights("models/plain_16_v5/1.654546069463094/weights.h5")

# Save the combined model as a single Keras HDF5 model file
model.save("models/plain_16_v5/1.654546069463094/combined_model.h5")