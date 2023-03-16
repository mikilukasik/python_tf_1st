from utils.train_model_v4 import train_model

# model_source = '../models/xs39_L_v1/_blank'
model_source = '../models/xs39_L_v1/1.8760964042345682'
model_dest = '../models/xs39_L_v1'

# initial_lr = 0.00001
# initial_lr = 0.0001
initial_batch_size = 64
lr_multiplier = 0.2


fixed_lr = 0.00001

#  initial
#  fixed_lr = 0.00005

xs_format = '39'

train_model(model_source, model_dest,
            initial_batch_size, gpu=True, lr_multiplier=lr_multiplier, fixed_lr=fixed_lr,  xs_format=xs_format)
