
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add, ReLU, Dropout
from keras.initializers import he_normal
from keras import regularizers
import numpy as np

from utils.load_model import load_model
from utils.save_model import save_model
from utils.create_champ_model import create_champ_model
import random
import sys

model_name = '../models/merged_trained_progtrain/1.6104516983032227_best'
model = load_model(model_name)


for layer in model.layers:
    print(layer.name)

    if layer.name == 'progress_model':
        # making progress_model non-trainable
        for sublayer in layer.layers:
            sublayer.trainable = False
    elif layer.name == 'opening_model':
        # making opening_model trainable
        for sublayer in layer.layers:
            sublayer.trainable = True
    elif layer.name == 'mid_end_model':
        # making mid_end_model trainable
        for sublayer in layer.layers:
            sublayer.trainable = True

model.summary()

save_model(model, '../models/merged_trained_progFixed/_orig')
