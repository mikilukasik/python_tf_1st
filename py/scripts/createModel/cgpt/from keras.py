from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, ELU, Add, Reshape, Permute, multiply

MODEL_NAME = '8L-64r2-128r2-256r2-512r2-K3-S2-P2-Arelu-D1024-D512-A1'

# In this format:
# 8L indicates that the network has 8 convolutional layers.
# 64r2-128r2-256r2-512r2 indicates the number of filters in each residual block and that each block has two layers.
# K3 indicates that the convolutional kernel size is 3x3.
# S2 indicates that the stride is 2x2.
# P2 indicates that the pooling size is 2x2.
# Arelu indicates that the activation function is ReLU.
# D1024-D512 indicates the two dense layers with 1024 and 512 units, respectively.
# A1 indicates that the model includes an attention block.

# Input layer with shape (8, 8, 14)
input_layer = Input(shape=(8, 8, 14))

# Two Conv2D layers with 64 filters each and ReLU activation
conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)
conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)

# Residual block with two Conv2D layers with 128 filters each and ReLU activation
res1 = Add()([conv1, conv2])
conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(res1)
conv4 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv3)

# Residual block with two Conv2D layers with 256 filters each and ReLU activation
res2 = Add()([conv3, conv4])
conv5 = Conv2D(256, (3, 3), padding='same', activation='relu')(res2)
conv6 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv5)

# Residual block with two Conv2D layers with 512 filters each and ReLU activation
res3 = Add()([conv5, conv6])
conv7 = Conv2D(512, (3, 3), padding='same', activation='relu')(res3)
conv8 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv7)

# Attention block
attention = Conv2D(512, (1, 1), activation='relu')(conv8)
attention = Conv2D(1, (1, 1), activation='sigmoid')(attention)
attention = Flatten()(attention)
attention = Dense(64, activation='softmax')(attention)
attention = Reshape((8, 8, 1))(attention)
attention = Permute((3, 1, 2))(attention)
conv8 = multiply([conv8, attention])

# MaxPooling layer
max_pool = MaxPooling2D(pool_size=(2, 2))(conv8)

# Flatten the input and the output of the final Conv2D layer
flatten_input = Flatten()(input_layer)
flatten_output = Flatten()(max_pool)

# Concatenate the flattened input and output
concat = Concatenate()([flatten_input, flatten_output])
