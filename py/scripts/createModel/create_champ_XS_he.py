from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add
from keras.initializers import he_normal
from utils.save_model import save_model

input = Input(shape=(8, 8, 14))

x = Conv2D(50, (3, 3), padding='same',
           kernel_initializer=he_normal(), activation='relu')(input)
y = Conv2D(50, (3, 3), padding='same',
           kernel_initializer=he_normal(), activation='relu')(x)
z = Add()([x, y])

x = Conv2D(100, (3, 3), padding='same',
           kernel_initializer=he_normal(), activation='relu')(z)
y = Conv2D(100, (3, 3), padding='same',
           kernel_initializer=he_normal(), activation='relu')(x)
z = Add()([x, y])

x = Conv2D(200, (3, 3), padding='same',
           kernel_initializer=he_normal(), activation='relu')(z)
y = Conv2D(200, (3, 3), padding='same',
           kernel_initializer=he_normal(), activation='relu')(x)
z = Add()([x, y])

flatInput = Flatten()(input)
flatConv = Flatten()(z)

concatted1 = Concatenate()([flatInput,  flatConv])
dense1 = Dense(540, kernel_initializer=he_normal(),
               activation='relu')(concatted1)
dense1 = ELU()(dense1)

concatted2 = Concatenate()([flatInput,  dense1])
dense2 = Dense(270, kernel_initializer=he_normal(),
               activation='relu')(concatted2)
dense2 = ELU()(dense2)

concatted3 = Concatenate()([flatInput,  dense2])

output = Dense(1837, activation='softmax')(concatted3)

model = Model(inputs=input, outputs=output)
model.summary()

save_model(model, '../models/champ_XS_he_v1/_blank')
