from utils.train_model_v4 import train_model

# model_source = '../models/tripple_assisted_v2/_blank'
model_source = '../models/tripple_assisted_v2/2.6479498545328775'
model_dest = '../models/tripple_assisted_v2'

initial_lr = 0.00001
# initial_lr = 0.0001
initial_batch_size = 256
lr_multiplier = 3

ys_format = '1966'

train_model(model_source, model_dest,
            initial_batch_size, initial_lr, gpu=True, lr_multiplier=lr_multiplier, ys_format=ys_format)
