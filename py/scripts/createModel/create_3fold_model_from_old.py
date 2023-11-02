import keras
from keras.layers import Input, Lambda
from keras.models import Model, clone_model

import tensorflow as tf

from utils.create_champ_model import create_champ_model
from utils.save_model import save_model
from utils.load_model import load_model


# Load the progress model (trined to predict if game is in opening stage)
progress_model_name = '../models/c16_32_64_skip_d22_dobc3_bn_o5/0.7432216770648957'
progress_model = load_model(progress_model_name)

# load the opening model (trained to predict the best move in the opening stage)
opening_model_name = '../models/inpConv_c16_64_128_skip_d25_do3_bn_upToProg20Winners_V3/1.5148398876190186_best'
opening_model = load_model(opening_model_name)

# load the mid/end model (trained to predict the best move in the mid/end game)
mid_end_model_name = '../models/allButOpenings_transf_inpConv_c16x2x6_skip_l2_d101010_l1_do1_bn/1.6221170337200164_temp'
mid_end_model = load_model(mid_end_model_name)

for layer in progress_model.layers:
    layer.trainable = False

model_1 = opening_model
model_2 = mid_end_model

model_3 = clone_model(mid_end_model)
model_3.set_weights(mid_end_model.get_weights())

# model_4 = clone_model(mid_end_model)
# model_4.set_weights(mid_end_model.get_weights())

# model_5 = clone_model(mid_end_model)
# model_5.set_weights(mid_end_model.get_weights())

progress_model._name = 'progress_model'
model_1._name = 'opening_model'
model_2._name = 'model_2'
model_3._name = 'model_3'
# model_4._name = 'model_4'
# model_5._name = 'model_5'


def merge_outputs(inputs):
    # Extract the model outputs
    output_1, output_2, output_3, ratios = inputs

    # Calculate merged output based on ratios
    final_output = (ratios[:, 0:1] * output_1 +
                    (ratios[:, 1:2]+ratios[:, 2:3]) * output_2 +
                    (ratios[:, 3:4]+ratios[:, 4:5]) * output_3)

    return final_output


# Input layer
input_tensor = Input(shape=(8, 8, 14))

# Getting outputs for all 6 models using the same input
output_1 = model_1(input_tensor)
output_2 = model_2(input_tensor)
output_3 = model_3(input_tensor)
# output_4 = model_4(input_tensor)
# output_5 = model_5(input_tensor)
ratios = progress_model(input_tensor)

# Merging the outputs
merged_output = Lambda(merge_outputs)(
    [output_1, output_2, output_3, ratios])

# Creating the merged model
merged_model = Model(inputs=input_tensor, outputs=merged_output)


merged_model.summary()

save_model(merged_model, '../models/merged_3models_fo/_orig')
