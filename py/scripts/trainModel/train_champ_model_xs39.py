from utils.train_model_v4 import train_model

# model_source = '../models/xs39_L_v2/_blank'
model_source = '../models/xs39_L_v2/2.0534344933827717'
model_dest = '../models/xs39_L_v2'

initial_batch_size = 64
lr_multiplier = 0.015


fixed_lr = 0.00003

#  initial
#  fixed_lr = 0.0003

xs_format = '39'

train_model(model_source, model_dest,
            initial_batch_size, gpu=True, lr_multiplier=lr_multiplier, fixed_lr=fixed_lr,  xs_format=xs_format)
