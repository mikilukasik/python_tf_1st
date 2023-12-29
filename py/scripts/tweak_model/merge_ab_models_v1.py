
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

model_a_name = '../models/new/c16to512AndBack/V1_allButOpenings/1.710611480951309 copy'
model_a = load_model(model_a_name)
model_a._name = 'new/c16to512AndBack/V1_allButOpenings/1.710611480951309'

model_b_name = '../models/new/c16to256AndBack/V1_allButOpenings/1.6833038201332093'
model_b = load_model(model_b_name)
model_b._name = 'new/c16to256AndBack/V1_allButOpenings/1.6833038201332093'

progress_model_name = '../models/c16_32_64_skip_d22_dobc3_bn_o1_v3/0.06691280452404605_temp'
progress_model = load_model(progress_model_name)
progress_model._name = 'c16_32_64_skip_d22_dobc3_bn_o1_v3/0.06691280452404605_temp'

opening_model_name = '../models/new/XL_p0_v2_do3/1.5499424990415571_temp'
opening_model = load_model(opening_model_name)
opening_model._name = 'new/XL_p0_v2_do3/1.5499424990415571_temp'


for layer in progress_model.layers:
    layer.trainable = False
for layer in opening_model.layers:
    layer.trainable = False


def merge_outputs(inputs):
    output_progress, output_opening, output_A, output_B = inputs

    mid_end = (output_A + output_B) / 2

    final_output = output_progress * output_opening + \
        (1 - output_progress) * mid_end

    return final_output


# Input layer
input_tensor = Input(shape=(8, 8, 14))

output_progress = progress_model(input_tensor)
output_opening = opening_model(input_tensor)
output_A = model_a(input_tensor)
output_B = model_b(input_tensor)


# Merging the outputs
merged_output = Lambda(merge_outputs)(
    [output_progress, output_opening, output_A, output_B])

# Creating the merged model
new_model = Model(inputs=input_tensor, outputs=merged_output)


new_model.summary()

# new_model.save('../models/merged_v1')
save_model(new_model, '../models/merged_ab_double_plus_opening/_orig')
