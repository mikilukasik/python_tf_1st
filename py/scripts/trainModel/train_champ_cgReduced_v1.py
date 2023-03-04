from utils.train_model import train_model

# with tf.device('/cpu:0'):

model_source = './models/champ_cgReduced_v1/_blank'
model_dest = './models/champ_cgReduced_v1'

learning_rate = 0.0003
BATCH_SIZE = 64

train_model(model_source, model_dest, BATCH_SIZE, learning_rate)
