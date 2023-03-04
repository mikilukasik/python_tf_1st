from utils.train_model import train_model

# with tf.device('/cpu:0'):

model_source = '../models/c4RESd2_S_v1/lastgood'
model_dest = '../models/c4RESd2_S_v1'

learning_rate = 0.0000003
BATCH_SIZE = 1024

train_model(model_source, model_dest, BATCH_SIZE, learning_rate)
