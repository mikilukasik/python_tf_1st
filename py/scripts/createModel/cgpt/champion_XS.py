from save_model import save_model
from create_chess_model import create_chess_model

MODEL_NAME = 'champion_XS'

input_shape = (8, 8, 14)
conv_blocks = [((3, 3), (16, 32, 64))]
activation = 'relu'
dense_units = (512, 256)
output_units = 1837

model = create_chess_model(model_name=MODEL_NAME,
                           input_shape=input_shape,
                           conv_blocks=conv_blocks,
                           activation=activation,
                           dense_units=dense_units,
                           output_units=output_units)

save_model(model, './models/' + MODEL_NAME + '_blank')
