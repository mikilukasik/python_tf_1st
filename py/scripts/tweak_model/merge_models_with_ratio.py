
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

# progress_model = create_champ_model(filter_nums=[16, 32, 64, 128, 256],
#                                     layers_per_conv_block=2,
#                                     dense_units=[512, 512],
#                                     layers_per_dense_block=1,
#                                     dropout_rate=0.25,
#                                     dropout_between_conv=False,
#                                     batch_normalization=True,
#                                     #  l2_reg=0.00001,
#                                     input_to_all_conv=True,
#                                     add_skip_connections=True,
#                                     out_units=1,
#                                     )

print('progress model layers:')
progress_model.summary()


# load the opening model (trained to predict the best move in the opening stage)

opening_model1_name = '../models/new/XL_p0_v2_do3/1.5499424990415571_temp'
opening_model2_name = '../models/inpConv_c16_64_128_skip_d25_do3_bn_upToProg20Winners_V3/1.5148398876190186_best'
opening_model1 = load_model(opening_model1_name)
opening_model2 = load_model(opening_model2_name)

# opening_model = create_champ_model(filter_nums=[16, 32, 64, 128, 256, 512],
#                                    layers_per_conv_block=2,
#                                    dense_units=[512, 512],
#                                    layers_per_dense_block=1,
#                                    dropout_rate=0.15,
#                                    dropout_between_conv=False,
#                                    batch_normalization=True,
#                                    #  l2_reg=0.00001,
#                                    input_to_all_conv=True,
#                                    add_skip_connections=True
#                                    )

# load the mid/end model (trained to predict the best move in the mid/end game)
mid_end_model1_name = '../models/new/XL_p1-4_mg4/1.5866281690075994_temp'
mid_end_model1 = load_model(mid_end_model1_name)

mid_end_model2_name = '../models/new/c16to512AndBack/V1_allButOpenings/1.5838702917099_best copy'
mid_end_model2 = load_model(mid_end_model2_name)

mid_end_model3_name = '../models/allButOpenings_transf_inpConv_c16x2x6_skip_l2_d101010_l1_do1_bn/1.6221170337200164_temp'
mid_end_model3 = load_model(mid_end_model3_name)

# Set all layers not trainable in all models except the progress model
models_to_make_non_trainable = [opening_model1,
                                opening_model2,
                                mid_end_model1, mid_end_model2, mid_end_model3]

for model in models_to_make_non_trainable:
    for layer in model.layers:
        layer.trainable = False

progress_model._name = 'progress_model'
opening_model1._name = 'opening_model1'
opening_model2._name = 'opening_model2'
mid_end_model1._name = 'mid_end_model1'
mid_end_model2._name = 'mid_end_model2'
mid_end_model3._name = 'mid_end_model3'


def merge_outputs(inputs):
    # output_A, output_B1, output_B2, ratio_C = inputs
    output_opening1, output_opening2, output_mid_end1, output_mid_end2, output_mid_end3, progress_ratio = inputs

    # Calculating the merged output
    final_output = progress_ratio * (output_opening1 + output_opening2) / 2 + \
        (1 - progress_ratio) * (output_mid_end1 +
                                output_mid_end2 + output_mid_end3) / 3

    # final_output = ratio_C * output_A + \
    #     (1 - ratio_C) * (output_B1 + output_B2) / 2

    return final_output


# Input layer
input_tensor = Input(shape=(8, 8, 14))

# Getting outputs for all 3 models using the same input
# output_A = opening_model(input_tensor)
# output_B1 = mid_end_model1(input_tensor)
# output_B2 = mid_end_model2(input_tensor)
# output_C = progress_model(input_tensor)


output_opening1 = opening_model1(input_tensor)
output_opening2 = opening_model2(input_tensor)
output_mid_end1 = mid_end_model1(input_tensor)
output_mid_end2 = mid_end_model2(input_tensor)
output_mid_end3 = mid_end_model3(input_tensor)
progress_ratio = progress_model(input_tensor)


# Merging the outputs
merged_output = Lambda(merge_outputs)(
    [output_opening1, output_opening2, output_mid_end1, output_mid_end2, output_mid_end3, progress_ratio])

# Creating the merged model
new_model = Model(inputs=input_tensor, outputs=merged_output)


new_model.summary()

# new_model.save('../models/merged_v1')
save_model(new_model, '../models/merged_2op3me/_orig')
