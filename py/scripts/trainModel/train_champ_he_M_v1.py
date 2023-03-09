from utils.train_model_v3 import train_model

# model_source = '../models/champ_he_M_v1/_blank'
model_source = '../models/champ_he_M_v1/1.671831710656484'
model_dest = '../models/champ_he_M_v1'

initial_lr = 5e-5 // 0.00005
initial_batch_size = 64

train_model(model_source, model_dest,
            initial_batch_size, initial_lr, gpu=False, force_lr=False)
