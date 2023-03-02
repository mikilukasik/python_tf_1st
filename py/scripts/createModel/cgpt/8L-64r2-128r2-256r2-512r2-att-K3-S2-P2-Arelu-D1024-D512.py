import sys
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, ELU, Add, Multiply, Reshape, Permute, Lambda

MODEL_NAME = '8L-64r2-128r2-256r2-512r2-att-K3-S2-P2-Arelu-D1024-D512'
# 8L: The network has 8 convolutional layers.
# 64r2-128r2-256r2-512r2-att: There are four residual blocks with two convolutional layers in each block, with 64, 128, 256, and 512 filters respectively. Additionally, there is an attention block.
# K3: The convolutional kernel size is 3x3.
# S2: The stride is 2x2.
# P2: The pooling size is 2x2.
# Arelu: The activation function is ReLU.
# D1024-D512: There are two dense layers with 1024 and 512 units, respectively.

# Input layer with shape (8, 8, 14)
input_layer = Input(shape=(8, 8, 14))


def residual_block(input_layer, filters):
    """Add a residual block with two convolutional layers to the model."""
    res = input_layer
    for _ in range(2):
        res = Conv2D(filters, (3, 3), padding='same', activation='relu')(res)
    output_block = Add()([input_layer, res])
    return output_block


# Two Conv2D layers with 64 filters each and ReLU activation
conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)
conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)

# Residual block with 128 filters
res1 = residual_block(conv2, 128)

# Residual block with 256 filters
res2 = residual_block(res1, 256)

# Residual block with 512 filters
res3 = residual_block(res2, 512)

# Attention block
att = Permute((3, 1, 2))(res3)
att = Reshape((512, 64))(att)
att = Dense(64, activation='softmax')(att)
att = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(att)
att = Permute((2, 1))(att)
att = Multiply()([res3, att])

# MaxPooling layer
max_pool = MaxPooling2D(pool_size=(2, 2))(att)

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

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

save_model(model, './models/blanks/' + MODEL_NAME)
