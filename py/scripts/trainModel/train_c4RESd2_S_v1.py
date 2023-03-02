# from utils.train_model import train_model

# fmt: off
import sys
sys.path.append('./py/utils')
from train_model import train_model
# fmt: on

# with tf.device('/cpu:0'):

model_source = ('./models/c4RESd2_S_v1/1.6885718894958495')
model_dest = './models/c4RESd2_S_v1'

learning_rate = 0.00003
BATCH_SIZE = 256

train_model(model_source, model_dest, BATCH_SIZE, learning_rate)
