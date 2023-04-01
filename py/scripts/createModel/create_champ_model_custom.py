
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add
from keras.initializers import he_normal
from helpers.create_champ_model import create_champ_model
from utils.save_model import save_model

# input = Input(shape=(8, 8, 39))

model = create_champ_model(filter_nums=[32, 82, 210, 537])
model.summary()

save_model(model, '../models/plain_32x2.56/_blank')
