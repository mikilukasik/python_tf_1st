from utils import train_model

# with tf.device('/cpu:0'):

model_source = './models/c4RESd2_S_v1/1.6441539738972981'
model_dest = './models/c4RESd2_S_v1'

learning_rate = 0.000001
BATCH_SIZE = 4096

train_model(model_source, model_dest, BATCH_SIZE, learning_rate)
