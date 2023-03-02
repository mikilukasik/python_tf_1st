from train_model import train_model


# with tf.device('/cpu:0'):

model_source = ('./models/champion_M/_blank')
model_dest = './models/champion_M'

learning_rate = 0.0001
BATCH_SIZE = 128

train_model(model_source, model_dest, BATCH_SIZE,
            learning_rate, from_json=True)
