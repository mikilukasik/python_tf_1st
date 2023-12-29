
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add, ReLU, Dropout
from keras.initializers import he_normal
from keras import regularizers
import numpy as np

from utils.load_model import load_model
from utils.save_model import save_model

from utils.create_champ_model import create_champ_model

import random

import sys

# Define the shared input layer
input_layer = Input(shape=(8, 8, 14))

# Load the progress model (trined to predict if game is in opening stage)
progress_model_name = '../models/c16_32_64_skip_d22_dobc3_bn_o1_v3/0.06691280452404605_temp'
progress_model = load_model(progress_model_name)

# load the opening model (trained to predict the best move in the opening stage)
opening_model_name = '../models/inpConv_c16_64_128_skip_d25_do3_bn_upToProg20Winners_V3/1.5148398876190186_best'
opening_model = load_model(opening_model_name)

# load the mid/end model (trained to predict the best move in the mid/end game)
mid_end_model_name = '../models/new/XL_p1-4_mg4/1.5905999898165464 copy'
mid_end_model = load_model(mid_end_model_name)


# Set all layers not trainable in all models

for layer in progress_model.layers:
    layer.trainable = False
for layer in opening_model.layers:
    layer.trainable = False
for layer in mid_end_model.layers:
    layer.trainable = False

print('progress model layers:')
progress_model.summary()


print('opening model layers:')
opening_model.summary()

print('mid/end model layers:')
mid_end_model.summary()

# progress_dense_layer_names = [layer.name for layer in progress_model.layers if isinstance(layer, Dense)]
# opening_dense_layer_names = [layer.name for layer in opening_model.layers if isinstance(layer, Dense)]
# mid_end_dense_layer_names = [layer.name for layer in mid_end_model.layers if isinstance(layer, Dense)]

dense_outputs_progress = [
    layer.output for layer in progress_model.layers if isinstance(layer, Dense)]
dense_outputs_opening = [
    layer.output for layer in opening_model.layers if isinstance(layer, Dense)]
dense_outputs_mid_end = [
    layer.output for layer in mid_end_model.layers if isinstance(layer, Dense)]

progress_model_multiple_outputs = Model(inputs=progress_model.input,
                                        outputs=[progress_model.output] + dense_outputs_progress)
opening_model_multiple_outputs = Model(inputs=opening_model.input,
                                       outputs=[opening_model.output] + dense_outputs_opening)
mid_end_model_multiple_outputs = Model(inputs=mid_end_model.input,
                                       outputs=[mid_end_model.output] + dense_outputs_mid_end)

progress_outputs = progress_model_multiple_outputs(input_layer)
dense_outputs_from_progess = progress_outputs[1:]
opening_outputs = opening_model_multiple_outputs(input_layer)
dense_outputs_from_opening = opening_outputs[1:]
mid_end_outputs = mid_end_model_multiple_outputs(input_layer)
dense_outputs_from_mid_end = mid_end_outputs[1:]

# progress_dense_outputs = [
#     layer.get_output_at(-1) for layer in progress_model.layers if isinstance(layer, Dense)]
# opening_dense_outputs = [
#     layer.get_output_at(-1) for layer in opening_model.layers if isinstance(layer, Dense)]
# mid_end_dense_outputs = [
#     layer.get_output_at(-1) for layer in mid_end_model.layers if isinstance(layer, Dense)]


# progress_model_connected = progress_model(input_layer)
# opening_model_connected = opening_model(input_layer)
# mid_end_model_connected = mid_end_model(input_layer)

# progress_dense_tensor_outputs = [tensor for tensor in progress_model_connected if tensor.name in [
#     output.name for output in progress_dense_outputs]]
# opening_dense_tensor_outputs = [tensor for tensor in opening_model_connected if tensor.name in [
#     output.name for output in opening_dense_outputs]]
# mid_end_dense_tensor_outputs = [tensor for tensor in mid_end_model_connected if tensor.name in [
#     output.name for output in mid_end_dense_outputs]]

flat_input = Flatten()(input_layer)
dense_layers = [flat_input] + dense_outputs_from_progess + \
    dense_outputs_from_opening + dense_outputs_from_mid_end

# # get all dense layers from the progress model
# for layer in progress_model.layers:
#     if isinstance(layer, Dense):
#         dense_layers.append(layer.output)

# # get all dense layers from the opening model
# for layer in opening_model.layers:
#     if isinstance(layer, Dense):
#         dense_layers.append(layer.output)

# # get all dense layers from the mid/end model
# for layer in mid_end_model.layers:
#     if isinstance(layer, Dense):
#         dense_layers.append(layer.output)


# concatenate all dense layers from all models, with a flattened input layer
concatted_dense = Concatenate(name='concat_dense_tw')(dense_layers)

# add 3 dense layers on top of the concatenated dense layers, 2048 2048 and 1024 units
x = Dense(4096, ELU(), kernel_initializer=he_normal())(concatted_dense)
x = Dropout(rate=0.1)(x)
x = Dense(2048, ELU(), kernel_initializer=he_normal())(Concatenate(name='concat' + str(random.randint(0, 999999))
                                                                   )([x, concatted_dense]))
x = Dropout(rate=0.1)(x)
x = Dense(1024, ELU(), kernel_initializer=he_normal())(Concatenate(name='concat' + str(random.randint(0, 999999))
                                                                   )([x, concatted_dense]))

x = Dropout(rate=0.1)(x)
# create an output layer with 1837 units and softmax activation
output_layer = Dense(1837, 'softmax', name="new_out")(Concatenate(name='concat' + str(random.randint(0, 999999))
                                                                  )([x, concatted_dense]))

# create a new model with the input layer and the output layer
new_model = Model(inputs=input_layer, outputs=output_layer)


new_model.summary()

# new_model.save('../models/merged_v1')
save_model(new_model, '../models/merged_L_v1/_blank')
