from utils.train_model_v3 import train_model

model_source = '../models/champ_XS_he_v1/_blank'
# model_source = '../models/champ_XS_he_v1/1.9247356957859463'
model_dest = '../models/champ_XS_he_v1'

# initial_lr = 0.0000003
initial_lr = 0.0005
initial_batch_size = 48

train_model(model_source, model_dest,
            initial_batch_size, initial_lr, gpu=False, force_lr=False)
