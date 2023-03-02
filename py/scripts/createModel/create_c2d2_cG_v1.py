
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate

input = Input(shape=(8, 8, 14))

conv3a = Conv2D(32, kernel_size=3, padding='same')(input)
d3a = ELU()(conv3a)

conv3b = Conv2D(64, kernel_size=3, padding='same')(d3a)
d3b = ELU()(conv3b)

conv8a = Conv2D(64, kernel_size=8, padding='same')(input)
d8a = ELU()(conv8a)

conv8b = Conv2D(128, kernel_size=8, padding='same')(d8a)
d8b = ELU()(conv8b)

flatInput = Flatten()(input)
flat3 = Flatten()(d3b)
flat8 = Flatten()(d8b)

concatted1 = Concatenate()([flatInput,  flat3, flat8])
dense1 = Dense(1024, ELU())(concatted1)

concatted2 = Concatenate()([flatInput,  dense1])
dense2 = Dense(512, ELU())(concatted2)

concatted3 = Concatenate()([flatInput,  dense2])

output = Dense(1837, 'softmax')(concatted3)

model = Model(inputs=input, outputs=output)
model.summary()
model.save('./models/blanks/c2d2_cG_v1')
