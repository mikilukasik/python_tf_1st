from utils.train_model_v3 import train_model

# model_source = '../models/champ_he_M_c5_v1/_blank'
model_source = '../models/champ_he_M_c5_v1/1.9706919725073708'
model_dest = '../models/champ_he_M_c5_v1'

initial_lr = 0.00001
# initial_lr = 0.0001
initial_batch_size = 64

train_model(model_source, model_dest,
            initial_batch_size, initial_lr, gpu=True, force_lr=True)
