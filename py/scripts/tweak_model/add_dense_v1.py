
from keras.models import Model
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add, ReLU, Dropout, Input, Lambda
from keras.initializers import he_normal
from keras import regularizers
import numpy as np

from utils.load_model import load_model
from utils.save_model import save_model

from utils.create_champ_model import create_champ_model

import random

import sys

# Define the shared input layer
# input_layer = Input(shape=(8, 8, 14))


# Load the progress model (trined to predict if game is in opening stage)
# progress_model_name = '../models/c16_32_64_skip_d22_dobc3_bn_o1_v3/0.06691280452404605_temp'
# progress_model = load_model(progress_model_name)

existing_model_name = '../models/new/c16to256AndBack/V1_allButOpenings/1.6833038201332093'
existing_model = load_model(existing_model_name)

print('existing model layers:')
existing_model.summary()

for layer in existing_model.layers:
    layer.trainable = False

# for i in range(len(existing_model.layers)):
#     print(i, existing_model.layers[i].name, existing_model.layers[i].trainable)


# Finding all the 'Add' layers in the model
add_layers_outputs = [
    layer.output for layer in existing_model.layers if isinstance(layer, Add)]

# flatten all those add layers
add_layers_outputs = [Flatten()(layer) for layer in add_layers_outputs]

# Adding the output of the flattened input layer to the list of outputs
# add_layers_outputs.append(existing_model.layers[81].output)

# Concatenating the outputs of all 'Add' layers
concatenated = Concatenate(
    name="all-the-adds-plus-flatinput")(add_layers_outputs) if add_layers_outputs else None

print('concatenated:', concatenated)

dense1 = Dense(512, activation=ELU())(concatenated)
dense2 = Dense(1024, activation=ELU())(dense1)
dense3 = Dense(512, activation=ELU())(dense2)

# Concatenating the last Dense layer with the concatenated output of the Add layers
# concat_with_last_dense = Concatenate()([dense3, concatenated])

# Adding the final output layer
output = Dense(1837, activation='softmax')(dense3)

# Creating the new model
new_model = Model(inputs=existing_model.input, outputs=output)

new_model.summary()

save_model(new_model, '../models/new_tweaked/16to256ab-Add_d5-10-5/_blank')


sys.exit()

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

# print('progress model layers:')
# progress_model.summary()


# load the opening model (trained to predict the best move in the opening stage)

opening_model_name = '../models/new/XL_p0_v2_do3/1.5499424990415571_temp'
# opening_model_name = '../models/inpConv_c16_64_128_skip_d25_do3_bn_upToProg20Winners_V3/1.5148398876190186_best'
opening_model = load_model(opening_model_name)

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
# mid_end_model_name = '../models/allButOpenings_transf_inpConv_c16x2x6_skip_l2_d101010_l1_do1_bn/1.6221170337200164_temp'
mid_end_model_name = '../models/new/XL_p1-4_mg4/1.5866281690075994_temp'
mid_end_model = load_model(mid_end_model_name)


# Set all layers not trainable in all models
for layer in progress_model.layers:
    layer.trainable = False
for layer in opening_model.layers:
    layer.trainable = False
# for layer in mid_end_model.layers:
#     layer.trainable = False

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
save_model(new_model, '../models/merged_XL3/_orig')
