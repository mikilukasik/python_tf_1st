from utils.train_model_v4 import train_model

model_source = '../models/champ_he_M_d600-d300_v1/_blank'
# model_source = '../models/champ_he_M_d600-d300_v1/2.2961974612715976'
model_dest = '../models/champ_he_M_d600-d300_v1'

initial_lr = 0.00001
# initial_lr = 0.0001
initial_batch_size = 64
lr_multiplier = 1

train_model(model_source, model_dest,
            initial_batch_size, initial_lr, gpu=True, lr_multiplier=lr_multiplier)

# model.save('../models/champ_he_M_d600-d300_v1/_blank')
