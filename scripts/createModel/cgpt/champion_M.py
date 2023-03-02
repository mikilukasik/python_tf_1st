from save_model import save_model
from create_chess_model import create_chess_model

MODEL_NAME = '2b-filters_64_128_256_512-dense_4096_2048-out1837'
# 2b- to indicate that it is a model with 2 convolutional blocks, followed by the filters and dense units, and the number of output units.

input_shape = (8, 8, 14)
conv_blocks = [((3, 3), (64, 128, 256, 512)),
               ((8, 8), (64, 128, 256, 512))]
activation = 'relu'
dense_units = (4096, 2048)
output_units = 1837

model = create_chess_model(model_name=MODEL_NAME,
                           input_shape=input_shape,
                           conv_blocks=conv_blocks,
                           activation=activation,
                           dense_units=dense_units,
                           output_units=output_units)

save_model(model, './models/blanks/' + MODEL_NAME)
