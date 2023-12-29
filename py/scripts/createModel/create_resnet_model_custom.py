
from utils.create_champ_model import create_champ_model
from utils.create_resnet import create_resnet
from utils.save_model import save_model


# Filter sequence as specified
filter_sequence = [16, 32, 64, 128, 256, 128, 64, 32, 16, 8]

# Create the ResNet model
model = create_resnet(input_shape=(
    8, 8, 14), num_classes=1837, filters_sequence=filter_sequence)
model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


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


# model = create_champ_model(filter_nums=[16, 32, 64, 128, 256, 128, 64, 32],
#                            layers_per_conv_block=2,
#                            # [512, 1024],  # [512, 1024],
#                            dense_units=[512, 1024, 512],
#                            layers_per_dense_block=1,

#                            #  dropout_rate=0.15,

#                            kernel_sizes=[3],
#                            #  dropout_between_conv=False,
#                            batch_normalization=True,
#                            #  l2_reg=0.00001,
#                            input_to_all_conv=True,
#                            add_skip_connections=True
#                            )
# # model_name = '../models/new/S_c16x2x5_bnorm_skip/_blank'
model_name = '../models/new_age/btlnck_resnet_v1/_blank'

model.summary()
save_model(model, model_name)
