import keras
from keras.models import model_from_json

# Load the model structure from the JSON file
with open("models/1.6615822938283287/model.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Load the model weights from the .h5 file
model.load_weights("models/1.6615822938283287/weights.h5")

# Save the combined model as a single Keras HDF5 model file
model.save("models/1.6615822938283287/combined_model.h5")