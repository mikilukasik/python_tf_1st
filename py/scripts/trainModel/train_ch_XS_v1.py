from utils.train_model_v4 import train_model

# model_source = '../models/ch_XS_v1/_blank'
model_source = '../models/ch_XS_v1_MESSUP/2.5198252204259237'
model_dest = '../models/ch_XS_v1_MESSUP'

initial_lr = 0.00001
# initial_lr = 0.0001
initial_batch_size = 64
lr_multiplier = 0.1

train_model(model_source, model_dest,
            initial_batch_size, initial_lr, gpu=True, lr_multiplier=lr_multiplier)
