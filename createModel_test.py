
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, LeakyReLU, Flatten, Concatenate

input = Input(shape=(8, 8, 14))

conv3a = Conv2D(128, kernel_size=3, activation=LeakyReLU(),
                padding='same')(input)
conv3b = Conv2D(256, kernel_size=3, activation=LeakyReLU(),
                padding='same')(conv3a)

conv8a = Conv2D(128, kernel_size=8, activation=LeakyReLU(),
                padding='same')(input)
conv8b = Conv2D(256, kernel_size=8, activation=LeakyReLU(),
                padding='same')(conv8a)

flatInput = Flatten()(input)
flat3 = Flatten()(conv3b)
flat8 = Flatten()(conv8b)
concatted = Concatenate()([flatInput, flat3, flat8])

dense = Dense(512, LeakyReLU())(concatted)

output = Dense(1837, 'softmax')(dense)

model = Model(inputs=input, outputs=output)

model.save('./1st_model')
