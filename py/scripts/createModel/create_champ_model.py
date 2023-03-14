
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add
from keras.initializers import he_normal
from helpers.create_champ_model import create_champ_model
from utils.save_model import save_model

model = create_champ_model(
    filter_nums=[32, 32, 32, 32], dense_units=[260, 130])
model.summary()
save_model(model, '../models/blk1_fn3-3-3-3_du26-13_v1/_blank')
