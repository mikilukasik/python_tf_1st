from utils.train_model_v4 import train_model

# model_source = '../models/champ_single_out_lin/_blank'
model_source = '../models/champ_single_out_lin_b8_v1/4.0460375695903545'
model_dest = '../models/champ_single_out_lin_b8_v1'

initial_batch_size = 256
lr_multiplier = 1
fixed_lr = 0.000001

train_model(model_source, model_dest,
            initial_batch_size, gpu=True, lr_multiplier=lr_multiplier, fixed_lr=fixed_lr, dataset_reader_version='17', filter='chkmtOrStallEndingOrHasBal8', ys_format='bal8')
