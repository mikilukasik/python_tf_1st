
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add, ReLU, Dropout
from keras.initializers import he_normal
from keras import regularizers
import numpy as np
import tensorflow as tf

from utils.load_model import load_model
from utils.save_model import save_model

from utils.create_champ_model import create_champ_model

import random

import sys


def rename_layers(model, prefix):
    """
    Rename the layers of the model by adding a prefix to each layer.
    """
    for layer in model.layers:
        layer._name = prefix + layer.name


# Define the shared input layer
input_layer = Input(shape=(8, 8, 14))

# Load the progress model (trined to predict if game is in opening stage)
progress_model_name = '../models/c16_32_64_skip_d22_dobc3_bn_o1_v3/0.06691280452404605_temp'
progress_model = load_model(progress_model_name)

# load the opening model (trained to predict the best move in the opening stage)
opening_model_name = '../models/inpConv_c16_64_128_skip_d25_do3_bn_upToProg20Winners_V3/1.5148398876190186_best'
opening_model = load_model(opening_model_name)

# load the mid/end model (trained to predict the best move in the mid/end game)
mid_end_model_name = '../models/allButOpenings_transf_inpConv_c16x2x6_skip_l2_d101010_l1_do1_bn/1.6221170337200164_temp'
mid_end_model = load_model(mid_end_model_name)


progress_model._name = 'progress_model'
opening_model._name = 'opening_model'
mid_end_model._name = 'mid_end_model'


# rename_layers(progress_model, "progress_")
# rename_layers(opening_model, "opening_")
# rename_layers(mid_end_model, "mid_end_")


# Set all layers not trainable in all models
for layer in progress_model.layers:
    layer.trainable = False
for layer in opening_model.layers:
    layer.trainable = False
for layer in mid_end_model.layers:
    layer.trainable = False

progress_output = progress_model(input_layer)
opening_output = opening_model(input_layer)
mid_end_output = mid_end_model(input_layer)

dense_layers = [progress_output, opening_output, mid_end_output]

concatted_dense = Concatenate(name='concat_dense_tw')(dense_layers)

x = Dense(2048, ELU(), kernel_initializer=he_normal())(concatted_dense)


output_layer = Dense(1837, 'softmax', name="new_out")(x)

# create a new model with the input layer and the output layer
new_model = Model(inputs=input_layer, outputs=output_layer)


new_model.summary()

# tf.saved_model.save(new_model, '../models/merged_S_v1/_blank')

# new_model.save('../models/merged_S_v1/_blank')
save_model(new_model, '../models/merged_S_v1/_blank')
