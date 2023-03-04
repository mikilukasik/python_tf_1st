from utils import train_model

# with tf.device('/cpu:0'):

model_source = '../models/nw_M_v1/2.8410207121299975'
# model_source = './models/nw_M_v1/_blank'
model_dest = '../models/nw_M_v1'

learning_rate = 0.00001
BATCH_SIZE = 64

train_model(model_source, model_dest, BATCH_SIZE, learning_rate)
