from utils.train_model_v4 import train_model

# model_source = '../models/plain_128x1.6/_blank'
model_source = '../models/plain_128x1.6_v1/1.7981926008860267'
model_dest = '../models/plain_128x1.6_v1'

initial_batch_size = 64
lr_multiplier = 1
fixed_lr = 0.00001

train_model(model_source, model_dest,
            initial_batch_size, gpu=True, lr_multiplier=lr_multiplier, fixed_lr=fixed_lr, dataset_reader_version='16')
