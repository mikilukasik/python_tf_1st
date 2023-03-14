from utils.train_model_v4 import train_model

# model_source = '../models/tripple_assisted_v2/_blank'
model_source = '../models/tripple_assisted_vx/2.867364841794211'
model_dest = '../models/tripple_assisted_vx'

initial_lr = 0.00001
# initial_lr = 0.0001
initial_batch_size = 64
lr_multiplier = 1
fixed_lr = 0.000005

ys_format = '1966'

train_model(model_source, model_dest,
            initial_batch_size, initial_lr, gpu=True, lr_multiplier=lr_multiplier, fixed_lr=fixed_lr, ys_format=ys_format)
