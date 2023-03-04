
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, LeakyReLU, Flatten, Concatenate

input = Input(shape=(8, 8, 14))

conv8a = Conv2D(256, kernel_size=8, padding='same', use_bias=False)(input)

conv8b = Conv2D(256, kernel_size=8, padding='same', use_bias=False)(conv8a)
d8b = LeakyReLU()(conv8b)

# conv8c = Conv2D(512, kernel_size=8, padding='same', use_bias=False)(d8b)
# d8c = LeakyReLU()(conv8c)

flatInput = Flatten()(input)
flat8 = Flatten()(conv8b)
concatted = Concatenate()([flatInput,  flat8])

dense = Dense(4096, LeakyReLU(), use_bias=False)(concatted)

output = Dense(1837, 'softmax', use_bias=False)(dense)

model = Model(inputs=input, outputs=output)
model.summary()
model.save('./models/blanks/c2d1_L_v1')
