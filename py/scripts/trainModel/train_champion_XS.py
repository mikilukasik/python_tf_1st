from train_model_v2 import train_model


# with tf.device('/cpu:0'):

# model_source = ('./models/champion_XS/_blank')
model_source = ('./models/champion_XS/3.458090090751648')
model_dest = './models/champion_XS'

learning_rate = 0.001
BATCH_SIZE = 32

train_model(model_source, model_dest, BATCH_SIZE,
            learning_rate, from_json=True)
