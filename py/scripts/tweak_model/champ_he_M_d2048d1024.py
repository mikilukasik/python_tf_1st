
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add, ReLU
from keras.initializers import he_normal
import numpy as np

from utils.load_model import load_model

# Load the saved model
model_name = '../models/champ_he_M_v1/1.6669645971722076'
model = load_model(model_name)

# Set all layers except the new dense layers to not trainable
for layer in model.layers:
    layer.trainable = False

model.summary()

for i in range(len(model.layers)):
    print(i, model.layers[i])


# Create new dense layers with double the units
new_dense1 = Dense(2048, ReLU(), kernel_initializer=he_normal())
new_dense2 = Dense(1024, ReLU(), kernel_initializer=he_normal())

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

x = new_dense1(model.layers[15].output)
x = new_dense2(Concatenate(name='concat_11')([model.layers[13].output, x]))

# output = model.layers[-1](Concatenate()([model.layers[13], x]))
output = Dense(1837, 'softmax')(Concatenate(
    name='concat_12')([model.layers[13].output, x]))


new_model = Model(inputs=input.output, outputs=output)

# Freeze all layers except the new dense layers
# for layer in new_model.layers[:-2]:
#     layer.trainable = False

# Print the updated model summary
new_model.summary()

model.save('../models/champ_he_MDD_v1/_blank')

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
