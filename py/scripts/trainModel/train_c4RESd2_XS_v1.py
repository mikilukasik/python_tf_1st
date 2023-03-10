from utils.train_model import train_model

# with tf.device('/cpu:0'):

model_source = './models/c4RESd2_XS_v1/2.1099851322174072'
model_dest = './models/c4RESd2_XS_v1'

learning_rate = 0.000003
BATCH_SIZE = 64

train_model(model_source, model_dest, BATCH_SIZE, learning_rate)
