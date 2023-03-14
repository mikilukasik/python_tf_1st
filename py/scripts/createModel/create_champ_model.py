
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add
from keras.initializers import he_normal
from helpers.create_champ_model import create_champ_model
from utils.save_model import save_model

model = create_champ_model(
    filter_nums=[40, 80, 160, 320], dense_units=[600, 300])
model.summary()
save_model(model, '../models/blk1_fn4-8-16-32_du60-30_v1/_blank')
