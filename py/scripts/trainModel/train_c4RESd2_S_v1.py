from utils.train_model import train_model

# with tf.device('/cpu:0'):

model_source = '../models/blanks/c4RESd2_S_v1'
# model_source = '../models/c4RESd2_S_v2_nextchamp/_blank'
model_dest = '../models/c4RESd2_S_v3_nextchamp'

# initial_learning_rate = 0.00003
initial_learning_rate = 0.0001

BATCH_SIZE = 1024
# BATCH_SIZE = 128

train_model(model_source, model_dest, BATCH_SIZE, initial_learning_rate)
