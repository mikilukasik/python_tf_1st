from utils.train_model_v4 import train_model

# model_source = '../models/champ_he_M_d1024-d1024_v1/_blank'
model_source = '../models/champ_he_M_d1024-d1024_v1/2.017127338886261'
model_dest = '../models/champ_he_M_d1024-d1024_v1'

initial_lr = 0.00001
# initial_lr = 0.0001
initial_batch_size = 64
lr_multiplier = .01

train_model(model_source, model_dest,
            initial_batch_size, initial_lr, gpu=True, lr_multiplier=lr_multiplier)
