
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add
from keras.initializers import he_normal
from helpers.create_champ_model import create_champ_model
from utils.save_model import save_model

# input = Input(shape=(8, 8, 39))

model = create_champ_model(out_units=1,out_activation='sigmoid')
model.summary()

save_model(model, '../models/champ_single_out/_blank')
