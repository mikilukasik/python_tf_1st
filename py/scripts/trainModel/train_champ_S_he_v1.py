from utils.train_model_v3 import train_model

# model_source = './models/champ_S_v1_he/_blank'
model_source = './models/champ_S_v2_he/1.9247356957859463'
model_dest = './models/champ_S_v2_he'

initial_lr = 0.0000003
# initial_lr = 0.0005
initial_batch_size = 32

train_model(model_source, model_dest,
            initial_batch_size, initial_lr, gpu=False, force_lr=True)
