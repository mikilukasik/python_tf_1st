
from utils.create_champ_model import create_champ_model
from utils.save_model import save_model

# model = create_champ_model(filter_nums=[16, 32, 64, 128], dropout_rate=0.2, dropout_between_conv=True, l2_reg=0.0001)
# model_name = '../models/plain_16x2x4_do2bc_l2r-4/_blank'

# model = create_champ_model(filter_nums=[16, 48, 144, 432], dense_units=[1024], dropout_rate=0.2, dropout_between_conv=True, l2_reg=0.0001)
# model_name = '../models/plain_16x3x4_d10_do2bc_l2r-4/_blank'

# model = create_champ_model(filter_nums=[16, 32, 64, 128, 256, 512], dense_units=[
#                            1024, 1024], dropout_rate=0.3, dropout_between_conv=False)
# model_name = '../models/plain_c16x2x6_d1010_do3/_blank'


# model = create_champ_model(filter_nums=[20, 40, 80, 160, 320, 640], dense_units=[
#                            1280, 1280], layers_per_dense_block=1,
#                            #  dropout_rate=0.05, dropout_between_conv=True,
#                            batch_normalization=True, l2_reg=0.00001)
# model_name = '../models/plain_c20x2x6_d1212_bnorm_l2-4/_blank'


model = create_champ_model(filter_nums=[16, 32, 64, 128, 256],
                           layers_per_conv_block=2,
                           dense_units=[512, 256],
                           layers_per_dense_block=1,
                           dropout_rate=0.2,
                           dropout_between_conv=False,
                           batch_normalization=True,
                           #  l2_reg=0.00001,
                           #  input_to_all_conv=True,
                           add_skip_connections=True,
                           out_units=5,
                           #  out_activation='sigmoid'
                           )
model_name = '../models/c16x2x5_skip_d52_do2_bn_o5/_blank'

model.summary()
save_model(model, model_name)
