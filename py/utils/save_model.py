import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from keras.utils.vis_utils import plot_model
from .print_large import print_large


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyJSONEncoder, self).default(obj)


def save_model(model, folder, metadata={}):
    """
    Saves a Keras model to disk, including the model architecture, its weights, the state of the optimizer,
    a summary of the model, a PNG visualization of the model with informative labels, and metadata.

    Parameters:
    - model: A Keras model object to be saved.
    - folder: A string representing the name of the directory to save the model files to.
    - metadata: A dictionary containing any additional metadata to be saved with the model.

    Returns:
    - None.
    """

    # Create directory for the model files
    os.makedirs(folder, exist_ok=True)

    # Save the complete model (architecture, weights, and optimizer state)
    model.save(os.path.join(folder, 'complete_model.h5'))

    # Save a summary of the model
    with open(os.path.join(folder, 'summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Create a PNG visualization of the model with informative labels
    plot_model(model, to_file=os.path.join(
        folder, 'model.png'), show_shapes=True)

    img = Image.open(os.path.join(folder, 'model.png'))
    draw = ImageDraw.Draw(img)

    # Add informative labels to the PNG visualization
    for layer in model.layers:
        if len(layer.output_shape) != 3:
            continue  # Skip layers with output_shape that is not a 3-tuple
        if layer.name.startswith('conv2d'):
            label = 'Conv2D'
        elif layer.name.startswith('max_pooling2d'):
            label = 'MaxPooling2D'
        elif layer.name.startswith('dense'):
            label = 'Dense'
        else:
            label = layer.name

        # Draw a label for the layer
        x = layer.output_shape[1] // 2
        y = layer.output_shape[2] // 2
        draw.text((x, y), label, fill=(255, 255, 255))

    img.save(os.path.join(folder, 'model.png'))

    # Save metadata
    with open(os.path.join(folder, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, cls=NumpyJSONEncoder)

    print_large('Model saved.', folder)
