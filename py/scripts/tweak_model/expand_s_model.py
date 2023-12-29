
from keras.models import Model
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add, ReLU, Dropout, Input,  Lambda

from keras.initializers import he_normal
from keras import regularizers
import numpy as np

from utils.load_model import load_model
from utils.save_model import save_model

from utils.create_champ_model import create_champ_model
import random
import sys


model_a_name = '../models/new/S_p1234_t1_c16_32_64_128_skip_d2_do1_bn/1.8580379486083984_best'
model_a = load_model(model_a_name)

model_a.summary()
for i in range(len(model_a.layers)):
    print(i, model_a.layers[i], model_a.layers[i].name)


model_a_p20to100 = Model(inputs=model_a.input,
                         outputs=model_a.layers[-4].output)
model_a_p20to100._name = 'model_a_p20to100'
model_a_p20to100.trainable = False
model_a_p20to100.summary()


model_b = create_champ_model(filter_nums=[16, 32, 64, 128, 256],
                             layers_per_conv_block=2,
                             dense_units=[512],
                             layers_per_dense_block=1,
                             dropout_rate=0.1,
                             dropout_between_conv=False,
                             batch_normalization=True,
                             #  l2_reg=0.00001,
                             input_to_all_conv=True,
                             add_skip_connections=True
                             )

model_b_p20to100 = Model(inputs=model_b.input,
                         outputs=model_b.layers[-4].output)
model_b_p20to100._name = 'model_b_p20to100'

# New shared input layer
new_input = Input(shape=(8, 8, 14))

# Ensure both models take the new input
output_a = model_a_p20to100(new_input)
output_b = model_b_p20to100(new_input)

# Flatten the new input and concatenate with the outputs of both submodels
flattened_input = Flatten()(new_input)
concatenated_output = Concatenate()([flattened_input, output_a, output_b])

dropped = Dropout(0.1)(concatenated_output)

new_output = Dense(1837, activation='softmax')(dropped)

combined_model = Model(inputs=new_input, outputs=new_output)


combined_model.summary()

# new_model.save('../models/merged_v1')
save_model(combined_model, '../models/new/comb2/_orig')
