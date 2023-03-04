
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, ELU, Add
from utils import save_model

MODEL_NAME = 'is-it-champ_v1'

# In this format:

# 8L indicates that the network has 8 convolutional layers.
# 64r2-128r2-256r2-512r2 indicates the number of filters in each residual block and that each block has two layers.
# K3 indicates that the convolutional kernel size is 3x3.
# S2 indicates that the stride is 2x2.
# P2 indicates that the pooling size is 2x2.
# Arelu indicates that the activation function is ReLU.
# D1024-D512 indicates the two dense layers with 1024 and 512 units, respectively.


# fmt: off
# import sys
# sys.path.append('./py/utils')
# fmt: on

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

# MaxPooling layer
max_pool = MaxPooling2D(pool_size=(2, 2))(conv8)

# Flatten the input and the output of the final Conv2D layer
flatten_input = Flatten()(input_layer)
flatten_output = Flatten()(max_pool)

# Concatenate the flattened input and output
concat = Concatenate()([flatten_input, flatten_output])

# Two Dense layers with 1024 and 512 units and ELU activation
dense1 = Dense(1024, activation='elu')(concat)
dense1 = Dropout(0.5)(dense1)
dense2 = Dense(512, activation='elu')(dense1)
dense2 = Dropout(0.5)(dense2)

# Concatenate the flattened input and the output of the last Dense layer
concat2 = Concatenate()([flatten_input, dense2])

# Output layer with a softmax activation
output_layer = Dense(1837, activation='softmax')(concat2)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

save_model(model, './models/' + MODEL_NAME + '/_blank')
