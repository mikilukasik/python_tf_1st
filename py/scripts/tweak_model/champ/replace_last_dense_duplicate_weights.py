
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add, ReLU
from keras.initializers import he_normal
import numpy as np
import tensorflow as tf
import numpy as np

from utils.load_model import load_model, load_model_meta
from utils.save_model import save_model


def extend_weights(_weights, target_shape=(1920, 1837)):
    weights = _weights[0]
    # Determine the amount of padding required for each dimension
    pad_width = [(0, target_shape[i] - weights.shape[i])
                 for i in range(len(target_shape))]

    # Pad the weights with zeros
    padded_weights = np.pad(weights, pad_width, mode='constant')

    return [padded_weights, _weights[1]]


def duplicate_weights(w):
    return [np.concatenate([w[0], w[0]], axis=1), np.concatenate([w[1], w[1]])]


def echo_weights(w):
    print(w)
    return [np.concatenate([w[0], w[0][896:]], axis=1), np.concatenate([w[1], w[1][896:]])]


def copy_weights(old_layer, new_layer, new_input_units, weight_extender=duplicate_weights):
    _ = new_layer(np.random.rand(10, new_input_units))

    print('old', old_layer.get_weights()[0].shape)
    print('new', new_layer.get_weights()[0].shape)

    old_weights = old_layer.get_weights()
    new_weights = weight_extender(old_weights)
    new_layer.set_weights(new_weights)


# Load the saved model
model_name = '../models/champ_he_M_v1/1.6645954327583314'
model = load_model(model_name)
model_meta = load_model_meta(model_name)

# Set all layers except the new dense layers to not trainable
for layer in model.layers:
    layer.trainable = False

model.summary()

for i in range(len(model.layers)):
    print(i, model.layers[i], model.layers[i].trainable)

input = model.layers[0]

new_layer = Dense(1024, ELU(),  name="dense2")
copy_weights(model.layers[18], new_layer, new_input_units=1920)

x = new_layer(model.layers[17].output)

# output = model.layers[-1](Concatenate()([model.layers[13], x]))
output_layer = Dense(1837, 'softmax', name="output")
copy_weights(model.layers[-1], output_layer,
             new_input_units=1920, weight_extender=extend_weights)


output = output_layer(Concatenate(
    name='concat_12')([model.layers[13].output, x]))


new_model = Model(inputs=input.output, outputs=output)

# Print the updated model summary
new_model.summary()
for i in range(len(new_model.layers)):
    print(i, new_model.layers[i], new_model.layers[i].trainable)

if not 'key_events' in model_meta:
    model_meta['key_events'] = []

model_meta['key_events'].append({
    'type': 'extend_dense_layer',
    'desc': 'last dense layer units extended from 512 to 1024, weights copied'
})

save_model(new_model, '../models/champ_he_M_d1024-d1024_v2/_blank', model_meta)
