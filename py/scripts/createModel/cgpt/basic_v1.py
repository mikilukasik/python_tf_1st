
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, ELU, Add
# The default parameters for the create_chess_model() function are as follows:

# input_shape is set to (8, 8, 14) which represents the shape of the input image.
# use_attention is set to False which means that the attention block is not used in the model.
# use_batch_norm is set to False which means that batch normalization is not used after each convolutional layer.
# use_global_avg_pool is set to False which means that global average pooling is not used before the final fully connected layer.
# conv_layers is set to 4 which represents the default number of convolutional layers in the model.
# filters is set to [64, 64, 128, 128] which represents the default number of filters for each convolutional layer.
# residual_blocks is set to 2 which represents the default number of residual blocks in the model.
# dense_layers is set to 2 which represents the default number of dense layers in the model.
# dense_units is set to [1024, 512] which represents the default number of units in each dense layer.
# loss is set to 'mse' which represents the default loss function to use in the model.
# activation is set to 'elu' which represents the default activation function to use in the model.
# Example usage of this function:

# python
# Copy code
# # Creating a basic chess model with default parameters
# model = create_chess_model()

# # Creating a chess model with attention block, batch normalization, and global average pooling
# model = create_chess_model(use_attention=True, use_batch_norm=True, use_global_avg_pool=True)

# # Creating a chess model with custom number of convolutional layers, residual blocks, and dense layers
# model = create_chess_model(conv_layers=6, residual_blocks=3, dense_layers=3)

# # Creating a chess model with custom loss and activation functions
# model = create_chess_model(loss='categorical_crossentropy', activation='softmax')
# In the format provided, the default parameters are represented as follows:

# python
# Copy code
MODEL_NAME = 'CL4-F64-64-128-128-R2-DL2-DU1024-512-Arelu-GBN0-GAP0-Att0'
# CL4 represents the default number of convolutional layers.
# F64-64-128-128 represents the default number of filters for each convolutional layer.
# R2 represents the default number of residual blocks.
# DL2-DU1024-512 represents the default number of dense layers and units.
# Arelu represents the default activation function.
# GBN0 represents that the default is not to use batch normalization.
# GAP0 represents that the default is not to use global average pooling.
# Att0 represents that the default is not to use the attention block.





# fmt: off
import sys
sys.path.append('./py/utils')
from save_model import save_model
from create_chess_model import create_chess_model
# fmt: on

model = create_chess_model()

save_model(model, './models/blanks/' + MODEL_NAME)
