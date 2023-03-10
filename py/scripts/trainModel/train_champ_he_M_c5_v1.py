from utils.train_model_v4 import train_model

# model_source = '../models/champ_he_M_c5_v1/_blank'
model_source = '../models/champ_he_M_c5_v1/1.8152797635155495'
model_dest = '../models/champ_he_M_c5_v1_MESSUP'

initial_lr = 0.00001
# initial_lr = 0.0001
initial_batch_size = 512

train_model(model_source, model_dest,
            initial_batch_size, initial_lr, gpu=True, force_lr=False)
