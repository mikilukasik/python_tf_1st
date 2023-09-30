from utils.train_model_v6 import train_model

# # model_source = '../models/plain_c16x2x5_d101010_do1bc/_blank'
model_source = '../models/_promising/plain_c16x2x5_d101010_do1bc_v2/1.767619252204895_best'
model_dest = '../models/plain_c16x2x5_d101010_do1bc_v3'


# # model_source = '../models/plain_c16x2x5_d1040_do1bc/_blank'
# model_source = '../models/plain_c16x2x5_d1040_do1bc/3.526905502591814'
# model_dest = '../models/plain_c16x2x5_d1040_do1bc'

# model_source = '../models/plain_c16x2x5_d10404020_do1bc/_blank'
# # model_source = '../models/plain_c16x2x5_d10404020_do1bc/3.526905502591814'
# model_dest = '../models/plain_c16x2x5_d10404020_do1bc'


# model_source = '../models/plain_c16x2x5_d101010_do1bc_mod2/_blank'
# # model_source = '../models/plain_c16x2x5_d101010_do1bc_mod2/3.526905502591814'
# model_dest = '../models/plain_c16x2x5_d101010_do1bc_mod2'


# model_source = '../models/plain_c16x2x6_d1010_do3/_blank'
# # model_source = '../models/plain_c16x2x6_d1010_do3/3.526905502591814'
# model_dest = '../models/plain_c16x2x6_d1010_do3'

# model_source = '../models/plain_c20x2x6_d1212_bnorm_l2-4/_blank'
# # model_source = '../models/plain_c20x2x6_d1212_bnorm_l2-4/1.8303914319276808_temp'
# model_dest = '../models/plain_c20x2x6_d1212_bnorm_l2-4'

# model_source = '../models/plain_c16x2x6_d0502_bnorm_l2-4/_blank'
# # model_source = '../models/plain_c16x2x6_d0502_bnorm_l2-4/1.8639376523494722'
# model_dest = '../models/plain_c16x2x6_d0502_bnorm_l2-4'


# # model_source = '../models/plain_c16x2x4_d_bnorm_l2-4/_blank'
# model_source = '../models/plain_c16x2x4_d_bnorm_l2-4/2.1682458033561707'
# model_dest = '../models/plain_c16x2x4_d_bnorm_l2-4'

# # model_source = '../models/plain_c16x2x6_d1010_bnorm_do3/_blank'
# model_source = '../models/plain_c16x2x6_d1010_bnorm_do3/2.1721010208129883_best'
# model_dest = '../models/plain_c16x2x6_d1010_bnorm_do3'


# # model_source = '../models/1.9572086135546367'
# model_source = '../models/large/1.8061499746392171_temp'
# model_dest = '../models/large'


# model_source = '../models/plain_c16x2x7_d105_do1_bnorm_l2-4/_blank'
# # model_source = '../models/plain_c16x2x7_d105_do1_bnorm_l2-4/1.8061499746392171_temp'
# model_dest = '../models/plain_c16x2x7_d105_do1_bnorm_l2-4'

# # model_source = '../models/plain_c16x2x4_lc4d2_d0505_do1_bnorm_l2-4/_blank'
# model_source = '../models/plain_c16x2x4_lc4d2_d0505_do1_bnorm_l2-4/2.4585278034210205_best'
# model_dest = '../models/plain_c16x2x4_lc4d2_d0505_do1_bnorm_l2-4'

# model_source = '../models/plain_16x2x6_do3/_blank'
# model_source = '../models/currchampv1/1.795102152943611_temp'
# model_dest = '../models/currchampv1'

# # model_source = '../models/inc_c16x2_l3_d256_l1_bn_l2-4/_blank'
# model_source = '../models/inc_c16x2_l3_d256_l1_bn_l2-4/2.4804459353993016'
# model_dest = '../models/inc_c16x2_l3_d256_l1_bn_l2-4'


# model_source = '../models/inpConv_c16x2x6_l2_d1010_l1_do1_bn_l2-4/_blank'
# # model_source = '../models/inpConv_c16x2x6_l2_d1010_l1_do1_bn_l2-4/2.4804459353993016'
# model_dest = '../models/inpConv_c16x2x6_l2_d1010_l1_do1_bn_l2-4'


# model_source = '../models/inpConv_c16x2x6_skip_l2_d1010_l1_do1_bn_l2-4/_blank'
# # model_source = '../models/inpConv_c16x2x6_skip_l2_d1010_l1_do1_bn_l2-4/2.4804459353993016'
# model_dest = '../models/inpConv_c16x2x6_skip_l2_d1010_l1_do1_bn_l2-4'


# model_source = '../models/inpConv_c16x2x5_skip_l2_d510_l1_do1_bn_l2-4/_blank'
# # model_source = '../models/inpConv_c16x2x5_skip_l2_d510_l1_do1_bn_l2-4/2.4804459353993016'
# model_dest = '../models/inpConv_c16x2x5_skip_l2_d510_l1_do1_bn_l2-4'


# # model_source = '../models/inpConv_c16x2x5_skip_l2_d510_l1_bn/_blank'
# model_source = '../models/inpConv_c16x2x5_skip_l2_d510_l1_bn/1.7406492165327072_temp'
# model_dest = '../models/inpConv_c16x2x5_skip_l2_d510_l1_bn'

# model_source = '../models/inpConv_c16x2x6_skip_l2_d510_l1_bn/_blank'
model_source = '../models/inpConv_c16x2x6_skip_l2_d510_l1_bn/1.7836655653715134'
model_dest = '../models/inpConv_c16x2x6_skip_l2_d510_l1_bn'


initial_batch_size = 256
lr_multiplier = 0.1  # 0.01  # 0.01
fixed_lr = 0.0002

train_model(model_source, model_dest,
            initial_batch_size, gpu=True, lr_multiplier=lr_multiplier, fixed_lr=fixed_lr,
            dataset_reader_version='18', filter='almostall', ys_format='default', make_trainable=True, evaluate=False)
