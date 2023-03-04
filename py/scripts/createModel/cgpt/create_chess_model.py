from typing import List, Tuple
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Conv2D, ELU, Flatten, Concatenate, Add
# from utils import save_model


def create_chess_model(model_name: str,
                       input_shape: Tuple[int, int, int] = (8, 8, 14),
                       conv_blocks: List[Tuple[Tuple[int, int], Tuple[int, ...]]] = [
                           ((3, 3), (64, 128, 256, 512))],
                       activation: str = 'relu',
                       dense_units: Tuple[int, int] = (1024, 512),
                       output_units: int = 1837) -> Model:
    """
    Creates a chess model with optional parameters.

    Parameters:
    - model_name: A string representing the name of the model.
    - input_shape: Tuple of 3 integers representing the shape of input data. Default is (8,8,14).
    - conv_blocks: List of tuples, where each tuple represents a convolutional block. 
      The tuple contains:
      - The size of the convolutional kernel.
      - The number of filters for each convolutional layer.
    - activation: String representing the activation function to use. Default is 'relu'.
    - dense_units: Tuple of 2 integers representing the number of units for dense layers. Default is (1024,512).
    - output_units: Integer representing the number of output units. Default is 1837.

    Returns:
    - A Keras model object.
    """

    input = Input(shape=input_shape)

    # Convolutional layers
    prev_outputs = []
    for i, block in enumerate(conv_blocks):
        kernel_size, filters = block
        num_layers = len(filters)
        prev_output = input
        for j in range(num_layers):
            x = Conv2D(filters[j], kernel_size, padding='same',
                       activation=activation)(prev_output)
            y = Conv2D(filters[j], kernel_size,
                       padding='same', activation=activation)(x)
            z = Add()([x, y])
            prev_output = z
        prev_outputs.append(prev_output)

    if len(prev_outputs) > 1:
        merged = Concatenate()(prev_outputs)
    else:
        merged = prev_outputs[0]

    flatInput = Flatten()(input)
    flatConv = Flatten()(merged)

    concatted1 = Concatenate()([flatInput, flatConv])
    dense1 = Dense(dense_units[0], ELU())(concatted1)

    concatted2 = Concatenate()([flatInput, dense1])
    dense2 = Dense(dense_units[1], ELU())(concatted2)

    concatted3 = Concatenate()([flatInput, dense2])

    output = Dense(output_units, 'softmax')(concatted3)

    model = Model(inputs=input, outputs=output)
    model.summary()

    # save_model(model, f'./models/{model_name}/_blank')

    return model
