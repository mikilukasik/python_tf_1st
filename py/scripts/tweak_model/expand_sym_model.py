from keras.models import Model
from keras.layers import Dense, Lambda, Input, Concatenate, Add, Flatten, Dropout


from utils.load_model import load_model
from utils.save_model import save_model

from utils.create_champ_model import create_champ_model

import sys


# Define the shared input layer


orig_model_name = '../models/new/c16to512AndBack/V1f_allButOpenings/1.5285521249040481'
orig_model = load_model(orig_model_name)

for layer in orig_model.layers:
    layer.trainable = False

input = orig_model.layers[0].output
flat_input = orig_model.layers[99].output
flattened_last_conv_output = orig_model.layers[100].output
# orig_model.summary()

# for i in range(len(orig_model.layers)):
#     print(i, orig_model.layers[i].name)

add_outs = [orig_model.layers[89].output, orig_model.layers[80].output,
            orig_model.layers[71].output, orig_model.layers[62].output, orig_model.layers[53].output]
flattened_add_outs = [Flatten()(add_out) for add_out in add_outs] + \
    [flattened_last_conv_output, orig_model.layers[-1].output]


concatted_add_outs = Concatenate()(
    flattened_add_outs)
# sys.exit()

x = Dense(1024, activation='relu')(concatted_add_outs)
x = Dropout(0.3)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(1837, activation='softmax')(Concatenate()([flat_input, x]))

# expander_model = create_champ_model(
#     input=input,
#     flat_input=flat_input,
#     filter_nums=[16, 32, 64, 128, 256],
#     layers_per_conv_block=2,
#     dense_units=[1024, 1024],
#     layers_per_dense_block=1,
#     dropout_rate=0.3,
#     kernel_sizes=[3],
#     batch_normalization=True,
#     input_to_all_conv=True,
#     add_skip_connections=True,
#     out_activation='linear',
#     out_units=1024,
#     out_kernel_initializer='zeros',
#     concat_to_flattened_conv=concatted_add_outs,
# )

# expander_model_output = expander_model.layers[-1].output


# Lambda function to combine outputs

# def combine_outputs(tensor):
#     pre_final, new_layers_output = tensor
#     return pre_final + new_layers_output


# combined_output = Lambda(combine_outputs)(
#     [flattened_last_conv_output, expander_model_output])

# concatted = Concatenate()([flat_input, combined_output])

# final_output = orig_model.layers[-1](concatted)

new_model = Model(inputs=orig_model.input, outputs=x)

new_model.summary()

save_model(new_model, '../models/new/c16to512AndBack/expanded_v4/_orig')
