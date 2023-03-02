from typing import Tuple
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add
from save_model import save_model


def create_chess_model(model_name: str,
                       input_shape: Tuple[int, int, int] = (8, 8, 14),
                       filters: Tuple[int, int, int, int] = (
                           64, 128, 256, 512),
                       dense_units: Tuple[int, int] = (1024, 512),
                       output_units: int = 1837) -> Model:
    """
    Creates a chess model with optional parameters.

    Parameters:
    - model_name: A string representing the name of the model.
    - input_shape: Tuple of 3 integers representing the shape of input data. Default is (8,8,14).
    - filters: Tuple of 4 integers representing the number of filters for convolutional layers. Default is (64,128,256,512).
    - dense_units: Tuple of 2 integers representing the number of units for dense layers. Default is (1024,512).
    - output_units: Integer representing the number of output units. Default is 1837.

    Returns:
    - A Keras model object.
    """

    input = Input(shape=input_shape)

    # Convolutional layers
    x = Conv2D(filters[0], (3, 3), padding='same', activation='relu')(input)
    y = Conv2D(filters[0], (3, 3), padding='same', activation='relu')(x)
    z = Add()([x, y])

    x = Conv2D(filters[1], (3, 3), padding='same', activation='relu')(z)
    y = Conv2D(filters[1], (3, 3), padding='same', activation='relu')(x)
    z = Add()([x, y])

    x = Conv2D(filters[2], (3, 3), padding='same', activation='relu')(z)
    y = Conv2D(filters[2], (3, 3), padding='same', activation='relu')(x)
    z = Add()([x, y])

    x = Conv2D(filters[3], (3, 3), padding='same', activation='relu')(z)
    y = Conv2D(filters[3], (3, 3), padding='same', activation='relu')(x)
    z = Add()([x, y])

    flatInput = Flatten()(input)
    flatConv = Flatten()(z)

    concatted1 = Concatenate()([flatInput,  flatConv])
    dense1 = Dense(dense_units[0], ELU())(concatted1)

    concatted2 = Concatenate()([flatInput,  dense1])
    dense2 = Dense(dense_units[1], ELU())(concatted2)

    concatted3 = Concatenate()([flatInput,  dense2])

    output = Dense(output_units, 'softmax')(concatted3)

    model = Model(inputs=input, outputs=output)
    model.summary()

    save_model(model, f'./models/{model_name}')

    return model
