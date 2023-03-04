from utils.train_model import train_model

# with tf.device('/cpu:0'):

model_source = './models/champ_cgReduced_v1/2.9880835355411874'
model_dest = './models/champ_cgReduced_v1'

learning_rate = 0.00003
BATCH_SIZE = 256

train_model(model_source, model_dest, BATCH_SIZE, learning_rate)
