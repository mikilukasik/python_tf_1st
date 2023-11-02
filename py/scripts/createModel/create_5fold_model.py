import keras
from keras.layers import Input, Lambda
from keras.models import Model
import tensorflow as tf

from utils.create_champ_model import create_champ_model
from utils.save_model import save_model
from utils.load_model import load_model


# Assuming model_1, model_2, ..., model_5, and model_Ratio are your pretrained models


# Load the progress model (trined to predict if game is in opening stage)
progress_model_name = '../models/c16_32_64_skip_d22_dobc3_bn_o5/0.7432216770648957'
progress_model = load_model(progress_model_name)


for layer in progress_model.layers:
    layer.trainable = False

model_1 = create_champ_model(filter_nums=[16, 32, 64, 128],
                             layers_per_conv_block=2,
                             dense_units=[256, 256],
                             layers_per_dense_block=1,
                             dropout_rate=0.2,
                             dropout_between_conv=False,
                             batch_normalization=True,
                             input_to_all_conv=True,
                             add_skip_connections=True,
                             )

model_2 = create_champ_model(filter_nums=[16, 32, 64, 128],
                             layers_per_conv_block=2,
                             dense_units=[256, 256],
                             layers_per_dense_block=1,
                             dropout_rate=0.2,
                             dropout_between_conv=False,
                             batch_normalization=True,
                             input_to_all_conv=True,
                             add_skip_connections=True,
                             )

model_3 = create_champ_model(filter_nums=[16, 32, 64, 128],
                             layers_per_conv_block=2,
                             dense_units=[256, 256],
                             layers_per_dense_block=1,
                             dropout_rate=0.2,
                             dropout_between_conv=False,
                             batch_normalization=True,
                             input_to_all_conv=True,
                             add_skip_connections=True,
                             )

model_4 = create_champ_model(filter_nums=[16, 32, 64, 128],
                             layers_per_conv_block=2,
                             dense_units=[256, 256],
                             layers_per_dense_block=1,
                             dropout_rate=0.2,
                             dropout_between_conv=False,
                             batch_normalization=True,
                             input_to_all_conv=True,
                             add_skip_connections=True,
                             )

model_5 = create_champ_model(filter_nums=[16, 32, 64, 128],
                             layers_per_conv_block=2,
                             dense_units=[256, 256],
                             layers_per_dense_block=1,
                             dropout_rate=0.2,
                             dropout_between_conv=False,
                             batch_normalization=True,
                             input_to_all_conv=True,
                             add_skip_connections=True,
                             )


model_6 = create_champ_model(filter_nums=[16, 32, 64, 128],
                             layers_per_conv_block=2,
                             dense_units=[256, 256],
                             layers_per_dense_block=1,
                             dropout_rate=0.2,
                             dropout_between_conv=False,
                             batch_normalization=True,
                             input_to_all_conv=True,
                             add_skip_connections=True,
                             )

progress_model._name = 'progress_model'
model_1._name = 'model_1'
model_2._name = 'model_2'
model_3._name = 'model_3'
model_4._name = 'model_4'
model_5._name = 'model_5'
model_6._name = 'model_6_shared'


def merge_outputs(inputs):
    # Extract the model outputs
    output_1, output_2, output_3, output_4, output_5, output_6, ratios = inputs

    # Calculate merged output based on ratios
    final_output = (ratios[:, 0:1] * output_1 +
                    ratios[:, 1:2] * output_2 +
                    ratios[:, 2:3] * output_3 +
                    ratios[:, 3:4] * output_4 +
                    ratios[:, 4:5] * output_5 + output_6)/2

    return final_output


# Input layer
input_tensor = Input(shape=(8, 8, 14))

# Getting outputs for all 6 models using the same input
output_1 = model_1(input_tensor)
output_2 = model_2(input_tensor)
output_3 = model_3(input_tensor)
output_4 = model_4(input_tensor)
output_5 = model_5(input_tensor)
output_6 = model_6(input_tensor)
ratios = progress_model(input_tensor)

# Merging the outputs
merged_output = Lambda(merge_outputs)(
    [output_1, output_2, output_3, output_4, output_5, output_6, ratios])

# Creating the merged model
merged_model = Model(inputs=input_tensor, outputs=merged_output)


merged_model.summary()

save_model(merged_model, '../models/5+1fold_XS/_orig')


# If you wish to compile and train or use the merged model, you can do so:
# merged_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# # model = create_champ_model(filter_nums=[16, 32, 64, 128], dropout_rate=0.2, dropout_between_conv=True, l2_reg=0.0001)
# # model_name = '../models/plain_16x2x4_do2bc_l2r-4/_blank'

# # model = create_champ_model(filter_nums=[16, 48, 144, 432], dense_units=[1024], dropout_rate=0.2, dropout_between_conv=True, l2_reg=0.0001)
# # model_name = '../models/plain_16x3x4_d10_do2bc_l2r-4/_blank'

# # model = create_champ_model(filter_nums=[16, 32, 64, 128, 256, 512], dense_units=[
# #                            1024, 1024], dropout_rate=0.3, dropout_between_conv=False)
# # model_name = '../models/plain_c16x2x6_d1010_do3/_blank'


# # model = create_champ_model(filter_nums=[20, 40, 80, 160, 320, 640], dense_units=[
# #                            1280, 1280], layers_per_dense_block=1,
# #                            #  dropout_rate=0.05, dropout_between_conv=True,
# #                            batch_normalization=True, l2_reg=0.00001)
# # model_name = '../models/plain_c20x2x6_d1212_bnorm_l2-4/_blank'


# model = create_champ_model(filter_nums=[16, 32, 64, 128, 256],
#                            layers_per_conv_block=2,
#                            dense_units=[512, 256],
#                            layers_per_dense_block=1,
#                            dropout_rate=0.2,
#                            dropout_between_conv=False,
#                            batch_normalization=True,
#                            #  l2_reg=0.00001,
#                            #  input_to_all_conv=True,
#                            add_skip_connections=True,
#                            out_units=5,
#                            #  out_activation='sigmoid'
#                            )
# model_name = '../models/c16x2x5_skip_d52_do2_bn_o5/_blank'

# model.summary()
# save_model(model, model_name)
