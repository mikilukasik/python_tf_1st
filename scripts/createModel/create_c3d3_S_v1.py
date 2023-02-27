
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, LeakyReLU, Flatten, Concatenate

input = Input(shape=(8, 8, 14))

conv3a = Conv2D(32, kernel_size=3, padding='same', use_bias=False)(input)
d3a = LeakyReLU()(conv3a)

conv3b = Conv2D(32, kernel_size=3, padding='same', use_bias=False)(d3a)
d3b = LeakyReLU()(conv3b)

conv3c = Conv2D(32, kernel_size=3, padding='same', use_bias=False)(d3b)
d3c = LeakyReLU()(conv3c)

conv8a = Conv2D(32, kernel_size=8, padding='same', use_bias=False)(input)
d8a = LeakyReLU()(conv8a)

conv8b = Conv2D(32, kernel_size=8, padding='same', use_bias=False)(d8a)
d8b = LeakyReLU()(conv8b)

conv8c = Conv2D(32, kernel_size=8, padding='same', use_bias=False)(d8b)
d8c = LeakyReLU()(conv8c)

flatInput = Flatten()(input)
flat3 = Flatten()(d3c)
flat8 = Flatten()(d8c)

concated1 = Concatenate()([flat3, flat8])
dense1 = Dense(1024, LeakyReLU(), use_bias=False)(concated1)
dense2 = Dense(512, LeakyReLU(), use_bias=False)(dense1)
dense3 = Dense(256, LeakyReLU(), use_bias=False)(dense2)

concated2 = Concatenate()([flatInput,  dense3])
output = Dense(1837, 'softmax', use_bias=False)(concated2)

model = Model(inputs=input, outputs=output)
model.summary()
model.save('./models/blanks/c3d3_S_v1')
