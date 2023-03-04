from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add, MaxPooling2D
from utils import save_model

input = Input(shape=(8, 8, 14))

x = Conv2D(32, (3, 3), padding='same', activation='relu')(input)
x = MaxPooling2D((2, 2))(x)
y = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
y = MaxPooling2D((2, 2))(y)

residual = Conv2D(128, (3, 3), padding='same')(y)
residual = ELU()(residual)
residual = Conv2D(128, (3, 3), padding='same')(residual)

z = Add()([residual, y])
z = Flatten()(z)

dense1 = Dense(256, activation='relu')(z)
dense2 = Dense(128, activation='relu')(dense1)

output = Dense(1837, activation='softmax')(dense2)

model = Model(inputs=input, outputs=output)
model.summary()
save_model(model, './models/champ_cgReduced_v1/_blank')
