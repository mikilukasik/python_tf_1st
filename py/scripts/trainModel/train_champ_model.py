from utils.train_model_v4 import train_model

model_source = '../models/blk1_fn4-8-16-32_du60-30_v1/_blank'
# model_source = '../models/blk1_fn4-8-16-32_du60-30_v2/2.8062087880240547'
model_dest = '../models/blk1_fn4-8-16-32_du60-30_v4'

initial_lr = 0.00001
# initial_lr = 0.0001
initial_batch_size = 64
lr_multiplier = 1
fixed_lr = 0.00005

train_model(model_source, model_dest,
            initial_batch_size, initial_lr, gpu=True, lr_multiplier=lr_multiplier, fixed_lr=fixed_lr)
