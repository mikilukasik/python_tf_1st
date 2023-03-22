from utils.train_model_v4 import train_model

# model_source = '../models/plain/_blank'
model_source = '../models/plain_16/1.8662850230711474'
model_dest = '../models/plain_16'

initial_batch_size = 64
lr_multiplier = 0.025
fixed_lr = 0.0003

train_model(model_source, model_dest,
            initial_batch_size, gpu=True, lr_multiplier=lr_multiplier, fixed_lr=fixed_lr, dataset_reader_version='16')
