from utils.train_model_v4 import train_model

# model_source = '../models/plain_16x2/_blank'
model_source = '../models/plain_16x2_v1_2700/2.099919319152832_best'
# model_source = '../models/plain_16x2_v1/0.8233004808425903_best'

model_dest = '../models/plain_16x2_v1_2700'

initial_batch_size = 128
lr_multiplier = 1
fixed_lr = 0.00003
# fixed_lr = 0.00000000003

train_model(model_source, model_dest,
            initial_batch_size, gpu=True, lr_multiplier=lr_multiplier, fixed_lr=fixed_lr, dataset_reader_version='18', filter='2700', ys_format='default')
