from utils.train_model import train_model

# with tf.device('/cpu:0'):

model_source = '../models/c4RESd2_S_v2_nextchamp/2.013741691907247'
model_dest = '../models/c4RESd2_S_v2_nextchamp'

# initial_learning_rate = 0.00003
initial_learning_rate = 0.00005

BATCH_SIZE = 2048
# BATCH_SIZE = 128

train_model(model_source, model_dest, BATCH_SIZE, initial_learning_rate)
