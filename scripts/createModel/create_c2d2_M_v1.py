
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, LeakyReLU, Flatten, Concatenate

input = Input(shape=(8, 8, 14))

conv3a = Conv2D(128, kernel_size=3, padding='same', use_bias=False)(input)
d3a = LeakyReLU()(conv3a)

conv3b = Conv2D(128, kernel_size=3, padding='same', use_bias=False)(d3a)
d3b = LeakyReLU()(conv3b)

conv8a = Conv2D(128, kernel_size=8, padding='same', use_bias=False)(input)
d8a = LeakyReLU()(conv8a)

conv8b = Conv2D(128, kernel_size=8, padding='same', use_bias=False)(d8a)
d8b = LeakyReLU()(conv8b)

flatInput = Flatten()(input)
flat3 = Flatten()(d3b)
flat8 = Flatten()(d8b)

concatted1 = Concatenate()([flatInput,  flat3, flat8])
dense1 = Dense(6000, LeakyReLU(), use_bias=False)(concatted1)

concatted2 = Concatenate()([flatInput,  dense1])
dense2 = Dense(4000, LeakyReLU(), use_bias=False)(concatted2)

concatted3 = Concatenate()([flatInput,  dense2])

output = Dense(1837, 'softmax', use_bias=False)(concatted3)

model = Model(inputs=input, outputs=output)
model.summary()
model.save('./models/blanks/c2d2_M_v1')
