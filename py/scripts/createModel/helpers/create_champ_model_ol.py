from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add, Dropout
from keras.initializers import he_normal
from keras.regularizers import l2
import random


def create_champ_model(dense_units=[1024, 512], filter_nums=[64, 128, 256, 512], kernel_size=3, layers_per_conv_block=2, conv_activation='relu', dense_activation=ELU(), kernel_initializer='he_normal', input=None, flat_input=None, return_dense_index=None, layer_name_prefix='', returned_dense_name=None, out_units=1837, out_activation='softmax', dropout_rate=0.0, dropout_between_conv=False, l2_reg=None):

    if input == None:
        input = Input(shape=(8, 8, 14))

    if flat_input == None:
        flat_input = Flatten(name=layer_name_prefix +
                             'Flatten-' + str(random.randint(0, 999999)))(input)

    x = input

    for i, num_filters in enumerate(filter_nums):
        layers_in_this_block = [
            Conv2D(num_filters, (kernel_size, kernel_size), padding='same', activation=conv_activation,
                   kernel_initializer=kernel_initializer, kernel_regularizer=l2(
                       l2_reg) if l2_reg else None,  # 3. Add regularization to Conv2D
                   name=layer_name_prefix + 'Conv2D-' + str(random.randint(0, 999999)))(x)
        ]

        for j in range(layers_per_conv_block - 1):
            next_conv_layer = Conv2D(num_filters, (kernel_size, kernel_size), padding='same', activation=conv_activation,
                                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg) if l2_reg else None, name=layer_name_prefix + 'Conv2D-' + str(random.randint(0, 999999)))(layers_in_this_block[-1])
            layers_in_this_block.append(next_conv_layer)

        x = Add(name=layer_name_prefix + 'Add-' +
                str(random.randint(0, 999999)))(layers_in_this_block)

        if dropout_rate > 0.0 and dropout_between_conv:
            x = Dropout(dropout_rate)(x)

    x = Flatten(name=layer_name_prefix + 'Flatten-' +
                str(random.randint(0, 999999)))(x)

    for i, num_units in enumerate(dense_units):
        x = Concatenate(name=layer_name_prefix + 'Concat-' +
                        str(random.randint(0, 999999)))([flat_input, x])
        x = Dense(num_units, activation=dense_activation,
                  kernel_initializer=kernel_initializer, kernel_regularizer=l2(
                      l2_reg) if l2_reg else None,
                  name=returned_dense_name if returned_dense_name is not None and return_dense_index is not None and i == return_dense_index
                  else layer_name_prefix + 'Dense-' + str(random.randint(0, 999999)))(x)
        if dropout_rate > 0.0:
            x = Dropout(dropout_rate)(x)

        if return_dense_index is not None and return_dense_index == i:
            return x

    x = Concatenate(name=layer_name_prefix + 'Dense-output-' +
                    str(random.randint(0, 999999)))([flat_input, x])
    output = Dense(out_units, activation=out_activation)(x)

    return Model(inputs=input, outputs=output)
