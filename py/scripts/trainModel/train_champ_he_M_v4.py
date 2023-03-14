from utils.train_model_v4 import train_model

# model_source = '../models/champ_he_M_v1/_blank'
model_source = '../models/champ_he_M_v1/1.6615822938283287'
model_dest = '../models/champ_he_M_v1'

initial_lr = 0.00001
# initial_lr = 0.0001
initial_batch_size = 64
lr_multiplier = 1
fixed_lr = 0.000001
# fixed_lr = 0.00005

train_model(model_source, model_dest,
            initial_batch_size, initial_lr, gpu=False, lr_multiplier=lr_multiplier, fixed_lr=fixed_lr)

# initial_lr = 5e-5  # 0.00005
# initial_batch_size = 64

# train_model(model_source, model_dest,
#             initial_batch_size, initial_lr, gpu=False, force_lr=False)
