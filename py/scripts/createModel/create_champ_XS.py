
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add
from utils import save_model

input = Input(shape=(8, 8, 14))

x = Conv2D(32, (3, 3), padding='same', activation='relu')(input)
y = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
z = Add()([x, y])

x = Conv2D(64, (3, 3), padding='same', activation='relu')(z)
y = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
z = Add()([x, y])

x = Conv2D(128, (3, 3), padding='same', activation='relu')(z)
y = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
z = Add()([x, y])

x = Conv2D(256, (3, 3), padding='same', activation='relu')(z)
y = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
z = Add()([x, y])

flatInput = Flatten()(input)
flatConv = Flatten()(z)

concatted1 = Concatenate()([flatInput,  flatConv])
dense1 = Dense(512, ELU())(concatted1)

concatted2 = Concatenate()([flatInput,  dense1])
dense2 = Dense(256, ELU())(concatted2)

concatted3 = Concatenate()([flatInput,  dense2])

output = Dense(1837, 'softmax')(concatted3)

model = Model(inputs=input, outputs=output)
model.summary()
save_model(model, './models/c4RESd2_XS_v1/_blank')
