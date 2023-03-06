from utils.train_model import train_model
import tensorflow as tf

model_source = './models/champ_S_v1_he/_blank'
# model_source = './models/champ_S_v1_he/3.234529365055145'
model_dest = './models/champ_S_v2_he'

initial_learning_rate = 0.0001

BATCH_SIZE = 64

with tf.device('/cpu:0'):
    train_model(model_source, model_dest, BATCH_SIZE, initial_learning_rate)
