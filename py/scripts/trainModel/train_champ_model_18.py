from utils.train_model_v4 import train_model

# model_source = '../models/champ_single_out_lin/_blank'
# model_source = '../models/champ_single_out_lin_nextbal14/2.930911832173665'
model_source = '../models/champ_single_out_lin_nextbal14/2.019265735149384_temp'



model_dest = '../models/champ_single_out_lin_nextbal14'

initial_batch_size = 256
lr_multiplier = 1
fixed_lr = 0.000001

train_model(model_source, model_dest,
            initial_batch_size, gpu=True, lr_multiplier=lr_multiplier, fixed_lr=fixed_lr, dataset_reader_version='18', filter='hasNextBal14', ys_format='nextBal14')

# hasAnyNextBal is another filter, might be better