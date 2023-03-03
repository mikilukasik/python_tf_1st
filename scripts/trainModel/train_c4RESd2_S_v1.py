from utils import train_model

# with tf.device('/cpu:0'):

model_source = './models/c4RESd2_S_v1/1.6460070133209228'
model_dest = './models/c4RESd2_S_v1'

learning_rate = 0.000003
BATCH_SIZE = 256

train_model(model_source, model_dest, BATCH_SIZE, learning_rate)
