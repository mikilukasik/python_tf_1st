import os
from keras.utils.vis_utils import plot_model
import json
from PIL import Image, ImageDraw, ImageFont


def save_model(model, filename, metadata={}):
    # Create directory for the model files
    os.makedirs(filename, exist_ok=True)

    # Save the model architecture as a JSON file
    with open(os.path.join(filename, 'model.json'), 'w') as f:
        f.write(model.to_json())

    # Save a summary of the model
    with open(os.path.join(filename, 'summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Create a PNG visualization of the model with informative labels
    plot_model(model, to_file=os.path.join(
        filename, 'model.png'), show_shapes=True)

    img = Image.open(os.path.join(filename, 'model.png'))

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
        draw.text((x, y), label, font=font, fill=(255, 255, 255))

    img.save(os.path.join(filename, 'model.png'))

    # Save metadata
    with open(os.path.join(filename, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
