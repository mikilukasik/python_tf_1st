from utils.train_model import train_model
import tensorflow as tf

# model_source = './models/champ_S_v1/_blank'
model_source = './models/champ_S_v1/3.234529365055145'
model_dest = './models/champ_S_v1'

initial_learning_rate = 0.0005

BATCH_SIZE = 2048

with tf.device('/cpu:0'):
    train_model(model_source, model_dest, BATCH_SIZE, initial_learning_rate)
