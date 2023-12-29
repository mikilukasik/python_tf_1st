from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add, Dropout, BatchNormalization, Activation
from keras.initializers import he_normal
from keras.regularizers import l2
import random


def create_champ_model(dense_units=[1024, 512], filter_nums=[64, 128, 256, 512], kernel_sizes=[3], layers_per_conv_block=2, layers_per_dense_block=1, conv_activation='relu', dense_activation=ELU(), kernel_initializer='he_normal', out_kernel_initializer=None, input=None, attach_cnn_to_layer=None, flat_input=None, return_dense_index=None, layer_name_prefix='', returned_dense_name=None, out_units=1837, out_activation='softmax', dropout_rate=0.0, dropout_between_conv=False, l2_reg=None, batch_normalization=False, concat_to_flattened_conv=None, input_to_all_conv=False, add_skip_connections=False):

    if input == None:
        input = Input(shape=(8, 8, 14))

    if flat_input == None:
        flat_input = Flatten(name=layer_name_prefix +
                             'Flatten-flat_input-' + str(random.randint(0, 999999)))(input)

    conv_outputs = []

    for kernel_size in kernel_sizes:

        x = attach_cnn_to_layer if attach_cnn_to_layer is not None else input

        for i, num_filters in enumerate(filter_nums):

            first_conv = Conv2D(num_filters, (kernel_size, kernel_size), padding='same',
                                kernel_initializer=kernel_initializer, kernel_regularizer=l2(
                l2_reg) if l2_reg else None,  # 3. Add regularization to Conv2D
                name=layer_name_prefix + 'Conv2D-conv0_in_res_block' + str(i) + '-' + str(random.randint(0, 999999)))(
                    Concatenate(name=layer_name_prefix + 'Concat-input_to_res_block' + str(i) + '-' + \
                                str(random.randint(0, 999999)))([x, input]) if (input_to_all_conv and i > 0) else x
            )

            if batch_normalization:
                first_conv = BatchNormalization(name=layer_name_prefix +
                                                'BatchNormalization-' + str(random.randint(0, 999999)))(first_conv)

            first_conv = Activation(conv_activation, name=layer_name_prefix +
                                    'Activation-' + str(random.randint(0, 999999)))(first_conv)

            layers_in_this_block = [
                (Conv2D(first_conv.shape[-1], (1, 1),
                        name=layer_name_prefix + 'Conv2D-skip_conn_transform_res_block' +
                        str(i) + '-' + str(random.randint(0, 999999))
                        )(
                    x)if x.shape[-1] != first_conv.shape[-1] else x),
                first_conv
            ] if add_skip_connections else [first_conv]

            for j in range(layers_per_conv_block - 1):
                next_conv_layer = Conv2D(num_filters, (kernel_size, kernel_size), padding='same',
                                         kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_reg) if l2_reg else None, name=layer_name_prefix + 'Conv2D-conv'+str(j+1) + '_in_res_block'+str(i) + '-' + str(random.randint(0, 999999)))(layers_in_this_block[-1])
                if batch_normalization:
                    next_conv_layer = BatchNormalization(name=layer_name_prefix +
                                                         'BatchNormalization-' + str(random.randint(0, 999999)))(next_conv_layer)

                next_conv_layer = Activation(conv_activation, name=layer_name_prefix +
                                             'Activation-' + str(random.randint(0, 999999)))(next_conv_layer)

                layers_in_this_block.append(next_conv_layer)

            x = Add(name=layer_name_prefix + 'Add-output_of_res_block' + str(i)+'-' +
                    str(random.randint(0, 999999)))(layers_in_this_block)

            if dropout_rate > 0.0 and dropout_between_conv:
                x = Dropout(dropout_rate, name='Dropout-' +
                            str(random.randint(0, 999999)))(x)

        if len(filter_nums) > 0:
            x = Flatten(name=layer_name_prefix + 'Flatten-flattened_last_conv-' +
                        str(random.randint(0, 999999)))(x)

        else:
            x = None

        if x is not None:
            conv_outputs.append(x)

    if len(filter_nums) > 0:
        if len(conv_outputs) > 1:
            x = Concatenate(name=layer_name_prefix + 'Concat-' +
                            str(random.randint(0, 999999)))(conv_outputs)
        else:
            x = conv_outputs[0]
    else:
        x = None

    if concat_to_flattened_conv is not None:
        x = Concatenate(name=layer_name_prefix + 'Concat-' +
                        str(random.randint(0, 999999)))([concat_to_flattened_conv, x])

    for i, num_units in enumerate(dense_units):
        x = flat_input if x is None else Concatenate(name=layer_name_prefix + 'Concat-' +
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
            x = Dropout(dropout_rate, name='Dropout-' +
                        str(random.randint(0, 999999)))(x)

    x = flat_input if x is None else Concatenate(name=layer_name_prefix + 'concat-' +
                                                 str(random.randint(0, 999999)))([flat_input, x])
    output = Dense(out_units, activation=out_activation, kernel_initializer=out_kernel_initializer if out_kernel_initializer is not None else kernel_initializer,
                   name=layer_name_prefix + 'Dense-output-' +
                   str(random.randint(0, 999999)))(x)

    return Model(inputs=input, outputs=output)
