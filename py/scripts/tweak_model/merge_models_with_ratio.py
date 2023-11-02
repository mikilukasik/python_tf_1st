
from keras.models import Model
from keras.layers import Input, Lambda
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
mid_end_model_name = '../models/allButOpenings_transf_inpConv_c16x2x6_skip_l2_d101010_l1_do1_bn/1.6221170337200164_temp'
mid_end_model = load_model(mid_end_model_name)


# Set all layers not trainable in all models
# for layer in progress_model.layers:
#     layer.trainable = False
for layer in opening_model.layers:
    layer.trainable = False
for layer in mid_end_model.layers:
    layer.trainable = False

progress_model._name = 'progress_model'
opening_model._name = 'opening_model'
mid_end_model._name = 'mid_end_model'


def merge_outputs(inputs):
    output_A, output_B, ratio_C = inputs

    # Calculating the merged output
    final_output = ratio_C * output_A + (1 - ratio_C) * output_B

    return final_output


# Input layer
input_tensor = Input(shape=(8, 8, 14))

# Getting outputs for all 3 models using the same input
output_A = opening_model(input_tensor)
output_B = mid_end_model(input_tensor)
output_C = progress_model(input_tensor)

# Merging the outputs
merged_output = Lambda(merge_outputs)([output_A, output_B, output_C])

# Creating the merged model
new_model = Model(inputs=input_tensor, outputs=merged_output)


new_model.summary()

# new_model.save('../models/merged_v1')
save_model(new_model, '../models/merged_trained_progtrain/_orig')
