from utils.train_model import train_model

# with tf.device('/cpu:0'):

model_source = '../models/champ_he_M_v1/_blank'
# model_source = '../models/c4RESd2_S_v2_nextchamp/_blank'
model_dest = '../models/champ_he_M_v1'

# initial_learning_rate = 0.00003
initial_learning_rate = 0.00003

BATCH_SIZE = 364
# BATCH_SIZE = 128

train_model(model_source, model_dest, BATCH_SIZE, initial_learning_rate)
