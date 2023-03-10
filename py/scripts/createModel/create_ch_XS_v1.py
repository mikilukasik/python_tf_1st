
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add
from keras.initializers import he_normal

input = Input(shape=(8, 8, 14))

x = Conv2D(64, (3, 3), padding='same', activation='relu',
           kernel_initializer=he_normal())(input)
y = Conv2D(64, (3, 3), padding='same', activation='relu',
           kernel_initializer=he_normal())(x)
z = Add()([x, y])

x = Conv2D(96, (3, 3), padding='same', activation='relu',
           kernel_initializer=he_normal())(z)
y = Conv2D(96, (3, 3), padding='same', activation='relu',
           kernel_initializer=he_normal())(x)
z = Add()([x, y])

x = Conv2D(144, (3, 3), padding='same', activation='relu',
           kernel_initializer=he_normal())(z)
y = Conv2D(144, (3, 3), padding='same', activation='relu',
           kernel_initializer=he_normal())(x)
z = Add()([x, y])

x = Conv2D(216, (3, 3), padding='same', activation='relu',
           kernel_initializer=he_normal())(z)
y = Conv2D(216, (3, 3), padding='same', activation='relu',
           kernel_initializer=he_normal())(x)
z = Add()([x, y])

flatInput = Flatten()(input)
flatConv = Flatten()(z)

concatted1 = Concatenate()([flatInput,  flatConv])
dense1 = Dense(768, ELU(), kernel_initializer=he_normal())(concatted1)

concatted2 = Concatenate()([flatInput,  dense1])
dense2 = Dense(384, ELU(), kernel_initializer=he_normal())(concatted2)

concatted3 = Concatenate()([flatInput,  dense2])

output = Dense(1837, 'softmax')(concatted3)

model = Model(inputs=input, outputs=output)
model.summary()
model.save('../models/ch_XS_v1/_blank')
