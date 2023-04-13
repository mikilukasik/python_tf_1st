from utils.train_model_v4 import train_model

# model_source = '../models/plain_32x3/_blank'
model_source = '../models/plain_32x3_v1/1.5069047991434734'
model_dest = '../models/plain_32x3_v1'

initial_batch_size = 64
lr_multiplier = 1
fixed_lr = 0.000003

train_model(model_source, model_dest,
            initial_batch_size, gpu=True, lr_multiplier=lr_multiplier, fixed_lr=fixed_lr, dataset_reader_version='16')
