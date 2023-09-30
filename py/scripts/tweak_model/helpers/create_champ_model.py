from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add, Dropout, BatchNormalization, Activation
from keras.initializers import he_normal
from keras.regularizers import l2
import random


def create_champ_model(dense_units=[1024, 512], filter_nums=[64, 128, 256, 512], kernel_size=3, layers_per_conv_block=2, layers_per_dense_block=1, conv_activation='relu', dense_activation=ELU(), kernel_initializer='he_normal', input=None, attach_cnn_to_layer=None, flat_input=None, return_dense_index=None, layer_name_prefix='', returned_dense_name=None, out_units=1837, out_activation='softmax', dropout_rate=0.0, dropout_between_conv=False, l2_reg=None, batch_normalization=False, concat_to_flattened_conv=None):

    if input == None:
        input = Input(shape=(8, 8, 14))

    if flat_input == None:
        flat_input = Flatten(name=layer_name_prefix +
                             'Flatten-' + str(random.randint(0, 999999)))(input)

    x = attach_cnn_to_layer.output if attach_cnn_to_layer is not None else input

    for i, num_filters in enumerate(filter_nums):

        first_conv = Conv2D(num_filters, (kernel_size, kernel_size), padding='same',
                            kernel_initializer=kernel_initializer, kernel_regularizer=l2(
            l2_reg) if l2_reg else None,  # 3. Add regularization to Conv2D
            name=layer_name_prefix + 'Conv2D-' + str(random.randint(0, 999999)))(x)

        if batch_normalization:
            first_conv = BatchNormalization(name=layer_name_prefix +
                                            'BatchNormalization-' + str(random.randint(0, 999999)))(first_conv)

        first_conv = Activation(conv_activation, name=layer_name_prefix +
                                'Activation-' + str(random.randint(0, 999999)))(first_conv)

        layers_in_this_block = [first_conv]

        for j in range(layers_per_conv_block - 1):
            next_conv_layer = Conv2D(num_filters, (kernel_size, kernel_size), padding='same',
                                     kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg) if l2_reg else None, name=layer_name_prefix + 'Conv2D-' + str(random.randint(0, 999999)))(layers_in_this_block[-1])
            if batch_normalization:
                next_conv_layer = BatchNormalization(name=layer_name_prefix +
                                                     'BatchNormalization-' + str(random.randint(0, 999999)))(next_conv_layer)

            next_conv_layer = Activation(conv_activation, name=layer_name_prefix +
                                         'Activation-' + str(random.randint(0, 999999)))(next_conv_layer)

            layers_in_this_block.append(next_conv_layer)

        x = Add(name=layer_name_prefix + 'Add-' +
                str(random.randint(0, 999999)))(layers_in_this_block)

        if dropout_rate > 0.0 and dropout_between_conv:
            x = Dropout(dropout_rate)(x)

    x = Flatten(name=layer_name_prefix + 'Flatten-' +
                str(random.randint(0, 999999)))(x)

    if concat_to_flattened_conv is not None:
        x = Concatenate(name=layer_name_prefix + 'Concat-' +
                        str(random.randint(0, 999999)))([concat_to_flattened_conv, x])

    for i, num_units in enumerate(dense_units):
        x = Concatenate(name=layer_name_prefix + 'Concat-' +
                        str(random.randint(0, 999999)))([flat_input, x])

        if layers_per_dense_block > 1:
            layers_in_this_block = [Dense(num_units, activation=dense_activation,
                                          kernel_initializer=kernel_initializer, kernel_regularizer=l2(
                                              l2_reg) if l2_reg else None,
                                          name=layer_name_prefix + 'Dense-' + str(random.randint(0, 999999)))(x)]
            for j in range(layers_per_dense_block - 1):
                next_dense_layer = Dense(num_units, activation=dense_activation,
                                         kernel_initializer=kernel_initializer, kernel_regularizer=l2(
                                             l2_reg) if l2_reg else None,
                                         name=returned_dense_name if returned_dense_name is not None and return_dense_index is not None and i == return_dense_index
                                         else layer_name_prefix + 'Dense-' + str(random.randint(0, 999999)))(layers_in_this_block[-1])
                layers_in_this_block.append(next_dense_layer)

            x = Add(name=returned_dense_name if returned_dense_name is not None and return_dense_index is not None and i == return_dense_index
                    else layer_name_prefix + 'Add-' +
                    str(random.randint(0, 999999)))(layers_in_this_block)
        else:
            x = Dense(num_units, activation=dense_activation,
                      kernel_initializer=kernel_initializer, kernel_regularizer=l2(
                          l2_reg) if l2_reg else None,
                      name=returned_dense_name if returned_dense_name is not None and return_dense_index is not None and i == return_dense_index else layer_name_prefix + 'Dense-' + str(random.randint(0, 999999)))(x)

        if return_dense_index is not None and return_dense_index == i:
            return x

        if dropout_rate > 0.0:
            x = Dropout(dropout_rate)(x)

    x = Concatenate(name=layer_name_prefix + 'Dense-output-' +
                    str(random.randint(0, 999999)))([flat_input, x])
    output = Dense(out_units, activation=out_activation)(x)

    return Model(inputs=input, outputs=output)
