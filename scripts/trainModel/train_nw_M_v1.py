from utils import train_model

# with tf.device('/cpu:0'):

model_source = './models/nw_M_v1/3.243776437441508'
# model_source = './models/nw_M_v1/_blank'
model_dest = './models/nw_M_v1'

learning_rate = 0.0001
BATCH_SIZE = 256

train_model(model_source, model_dest, BATCH_SIZE, learning_rate)
