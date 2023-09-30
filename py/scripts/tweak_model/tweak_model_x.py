
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add, ReLU, Dropout
from keras.initializers import he_normal
from keras import regularizers
import numpy as np

from utils.load_model import load_model

from utils.create_champ_model import create_champ_model

import random

import sys


# Load the saved model
model_name = '../models/plain_c16x2x2_d_bnorm_l2-4/2.2979659686088563'
model = load_model(model_name)

# Set all layers except the new dense layers to not trainable
for layer in model.layers:
    layer.trainable = False

model.summary()

for i in range(len(model.layers)):
    print(i, model.layers[i], model.layers[i].name)

# sys.exit()

# Create new dense layers with double the units
# new_dense1 = Dense(1024, ELU(), kernel_initializer=he_normal(), activity_regularizer=regularizers.l2(0.001))


# # Initialize the weights of the new layers using the weights of the old layers
# old_weights1 = model.layers[8].get_weights()
# new_weights1 = [np.concatenate([old_weights1[0], old_weights1[0]], axis=1), np.concatenate([
#     old_weights1[1], old_weights1[1]])]
# new_dense1.set_weights(new_weights1)

# old_weights2 = model.layers[10].get_weights()
# new_weights2 = [np.concatenate([old_weights2[0], old_weights2[0]], axis=1), np.concatenate([
#     old_weights2[1], old_weights2[1]])]
# new_dense2.set_weights(new_weights2)

# Create a new model with the updated architecture
input = model.layers[0]

flat_input = model.layers[12]
flattened_conv1 = model.layers[13]
flattened_conv2 = model.layers[20]
last_activated_conv = model.layers[18]
latest_softmax = model.layers[23]
# first_dense = model.layers[39]
# second_dense = model.layers[41]

new_model = create_champ_model(input=input.output,
                               attach_cnn_to_layer=last_activated_conv,
                               flat_input=flat_input.output,
                               filter_nums=[64],
                               dense_units=[],
                               layers_per_dense_block=1,
                               #  dropout_rate=0.05, dropout_between_conv=True,
                               batch_normalization=True,
                               l2_reg=0.00001,
                               concat_to_flattened_conv=Concatenate(name='concat-' + str(random.randint(0, 999999)))([
                                   latest_softmax.output,
                                   flattened_conv1.output,
                                   flattened_conv2.output,
                               ]))
# old_dense1 = model.layers[24]
# old_dense2 = model.layers[27]
# old_dense3 = model.layers[30]
# old_out = model.layers[33]

# first_dense_concat = model.layers[25]

# concatted_old_layers = Concatenate(name='concat_old_dense')([
#     flatInput.output,
#     old_dense1.output,
#     old_dense2.output,
#     old_dense3.output,
#     old_out.output
# ])

# new_dense1 = Dense(2048, ELU(), kernel_initializer=he_normal())
# new_dense2 = Dense(2048, ELU(), kernel_initializer=he_normal())


# input = Input(shape=(8, 8, 14))

# x = model.layers[1](input)
# model.layers[13] = Flatten()(input)

# for i in range(0, 12):
#     x = model.layers[i](x)

# x = Add()([model.layers[7].output, model.layers[8].output])
# x = Add()([model.layers[9].output, model.layers[10].output])
# x = Add()([model.layers[11].output, model.layers[12].output])
# x = Add()([model.layers[13].output, model.layers[14].output])
# flatInput = Flatten()(input)
# x = Flatten()(x)

# x = Dense(1024, ELU(), kernel_initializer=he_normal(),
#           name="new_dense1")(first_dense_concat.output)
# # x = Dense(2048, ELU(), kernel_initializer=he_normal(), name="new_dense2")(
# #     Concatenate(name='concat_x2')([concatted_old_layers, x]))
# x = Dropout(rate=0.3, name="do_1")(x)
# # output = model.layers[-1](Concatenate()([model.layers[13], x]))
# output = Dense(1837, 'softmax', name="new_out")(Concatenate(
#     name='concat_13')([flatInput.output, x]))


# new_model = Model(inputs=input.output, outputs=output)

# Freeze all layers except the new dense layers
# for layer in new_model.layers[:-2]:
#     layer.trainable = False

# Print the updated model summary
new_model.summary()
# for i in range(len(new_model.layers)):
#     print(i, new_model.layers[i], new_model.layers[i].trainable)

new_model.save('../models/plain_c16x2x3_d_bnorm_l2-4/_blank')

# # Set all layers except the new dense layers to not trainable
# for layer in model.layers:
#     layer.trainable = False

# # Set the new dense layers to trainable
# new_dense1.trainable = True
# new_dense2.trainable = True

# # # Set the output layer to not trainable
# # model.layers[-1].trainable = False

# # Replace the old dense layers with the new ones
# model.layers[16] = new_dense1
# model.layers[18] = new_dense2

# # Print the updated model summary
# model.summary()
