from utils.train_model import train_model

# with tf.device('/cpu:0'):

model_source = ('./models/blanks/c3RESd2_S_v1')
model_dest = './models/c3RESd2_S_v1'

learning_rate = 0.0003
BATCH_SIZE = 256

train_model(model_source, model_dest, BATCH_SIZE, learning_rate)
