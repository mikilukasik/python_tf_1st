from utils import train_model

# with tf.device('/cpu:0'):

model_source = './models/c4RESd2_S_v1/1.6588655837376913_temp'
model_dest = './models/c4RESd2_S_v1'

learning_rate = 0.00001
BATCH_SIZE = 2048

train_model(model_source, model_dest, BATCH_SIZE, learning_rate)
