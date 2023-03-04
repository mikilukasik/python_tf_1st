from helpers.create_chess_model import create_chess_model
from helpers.save_model import save_model

MODEL_NAME = 'nw_M_v1'

model = create_chess_model()
# conv_filters=[128], residual_filters=[128, 256, 512, 1024], use_attention=False)
model.summary()

save_model(model, './models/' + MODEL_NAME + '/_blank')
