from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add
from keras.initializers import he_normal


def create_champ_model(dense_units=[1024, 512], filter_nums=[64, 128, 256, 512], kernel_size=3, layers_per_conv_block=2, conv_activation='relu', dense_activation=ELU(), kernel_initializer=he_normal()):
    input = Input(shape=(8, 8, 14))
    flatInput = Flatten()(input)

    x = input

    for i, num_filters in enumerate(filter_nums):
        layers_in_this_block = [
            Conv2D(num_filters, (kernel_size, kernel_size), padding='same', activation=conv_activation,
                   kernel_initializer=kernel_initializer)(x)
        ]

        for j in range(layers_per_conv_block - 1):
            next_conv_layer = Conv2D(num_filters, (kernel_size, kernel_size), padding='same', activation=conv_activation,
                                     kernel_initializer=kernel_initializer)(layers_in_this_block[-1])
            layers_in_this_block.append(next_conv_layer)

        x = Add()(layers_in_this_block)

    x = Flatten()(x)
    x = Concatenate()([flatInput, x])

    for i, num_units in enumerate(dense_units):
        x = Dense(num_units, dense_activation,
                  kernel_initializer=kernel_initializer)(x)
        x = Concatenate()([flatInput, x])

    output = Dense(1837, 'softmax')(x)

    return Model(inputs=input, outputs=output)
