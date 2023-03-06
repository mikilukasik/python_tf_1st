from utils.train_model_v2 import train_model_v2
import tensorflow as tf

# model_source = './models/champ_S_v1_he/_blank'
model_source = './models/champ_S_v2_he/2.2293502078413154'
model_dest = './models/champ_S_v2_he'

initial_lr = 0.00005
initial_batch_size = 64

with tf.device('/cpu:0'):
    train_model_v2(model_source, model_dest,
                   initial_batch_size, initial_lr)
