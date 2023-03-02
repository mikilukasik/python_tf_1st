from save_model import save_model
from create_chess_model import create_chess_model

MODEL_NAME = '4r-filters_64_128_256_512-dense_1024_512-out1837'
# This filename representation includes the default values for the filters, dense_units, and output_units parameters in the function definition. It replaces the square brackets with underscores and separates the values with underscores as well, making it a good format for a filename. It also includes a prefix of "4r-" to indicate that this is a saved 4-block residual model. Finally, it uses an abbreviated form of "dense_units" and "output_units" to save space.

model_name = 'my_model'
input_shape = (8, 8, 14)
conv_blocks = [(4, (3, 3), (64, 128, 256, 512)),
               (3, (8, 8), (64, 128, 256))]
activation = 'relu'
dense_units = (1024, 512)
output_units = 1837

model = create_chess_model(model_name=model_name,
                           input_shape=input_shape,
                           conv_blocks=conv_blocks,
                           activation=activation,
                           dense_units=dense_units,
                           output_units=output_units)

save_model(model, './models/blanks/' + MODEL_NAME)
