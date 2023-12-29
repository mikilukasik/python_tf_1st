
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


model_a_name = '../models/merged_XL3_train_midend2/1.6168354749679565_best'
model_a = load_model(model_a_name)
model_a._name = 'merged_XL3_train_midend2/1.6168354749679565_best'

model_b_name = '../models/merged_trained_progFixed/1.596966028213501_best copy'
model_b = load_model(model_b_name)
model_b._name = 'merged_trained_progFixed/1.596966028213501_best'

model_c_name = '../models/new/c16to512AndBack/V1_allButOpenings/1.710611480951309 copy'
model_c = load_model(model_c_name)
model_c._name = 'new/c16to512AndBack/V1_allButOpenings/1.6003290232419969'


# # Load the progress model (trined to predict if game is in opening stage)
# progress_model_name = '../models/c16_32_64_skip_d22_dobc3_bn_o1_v3/0.06691280452404605_temp'
# progress_model = load_model(progress_model_name)

# # progress_model = create_champ_model(filter_nums=[16, 32, 64, 128, 256],
# #                                     layers_per_conv_block=2,
# #                                     dense_units=[512, 512],
# #                                     layers_per_dense_block=1,
# #                                     dropout_rate=0.25,
# #                                     dropout_between_conv=False,
# #                                     batch_normalization=True,
# #                                     #  l2_reg=0.00001,
# #                                     input_to_all_conv=True,
# #                                     add_skip_connections=True,
# #                                     out_units=1,
# #                                     )

# print('progress model layers:')
# progress_model.summary()


# # load the opening model (trained to predict the best move in the opening stage)

# opening_model_name = '../models/new/XL_p0_v2_do3/1.5499424990415571_temp'
# # opening_model_name = '../models/inpConv_c16_64_128_skip_d25_do3_bn_upToProg20Winners_V3/1.5148398876190186_best'
# opening_model = load_model(opening_model_name)

# # opening_model = create_champ_model(filter_nums=[16, 32, 64, 128, 256, 512],
# #                                    layers_per_conv_block=2,
# #                                    dense_units=[512, 512],
# #                                    layers_per_dense_block=1,
# #                                    dropout_rate=0.15,
# #                                    dropout_between_conv=False,
# #                                    batch_normalization=True,
# #                                    #  l2_reg=0.00001,
# #                                    input_to_all_conv=True,
# #                                    add_skip_connections=True
# #                                    )

# # load the mid/end model (trained to predict the best move in the mid/end game)
# # mid_end_model_name = '../models/allButOpenings_transf_inpConv_c16x2x6_skip_l2_d101010_l1_do1_bn/1.6221170337200164_temp'
# mid_end_model_name = '../models/new/XL_p1-4_mg4/1.5866281690075994_temp'
# mid_end_model = load_model(mid_end_model_name)


# Set all layers not trainable in all models
# for layer in progress_model.layers:
#     layer.trainable = False
# for layer in opening_model.layers:
#     layer.trainable = False
# for layer in mid_end_model.layers:
#     layer.trainable = False

# progress_model._name = 'progress_model'
# opening_model._name = 'opening_model'
# mid_end_model._name = 'mid_end_model'


def merge_outputs(inputs):
    output_A, output_B, output_C = inputs

    # Calculating the merged output
    final_output = (output_A + output_B + output_C) / 3

    return final_output


# Input layer
input_tensor = Input(shape=(8, 8, 14))

# Getting outputs for all 3 models using the same input
output_A = model_a(input_tensor)
output_B = model_b(input_tensor)
output_C = model_c(input_tensor)

# Merging the outputs
merged_output = Lambda(merge_outputs)([output_A, output_B, output_C])

# Creating the merged model
new_model = Model(inputs=input_tensor, outputs=merged_output)


new_model.summary()

# new_model.save('../models/merged_v1')
save_model(new_model, '../models/merged_triple_v2/_orig')
