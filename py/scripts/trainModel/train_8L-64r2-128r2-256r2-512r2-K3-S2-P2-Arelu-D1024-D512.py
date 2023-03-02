# from utils.train_model import train_model

# fmt: off
import sys
sys.path.append('./py/utils')
from train_model import train_model
# fmt: on

# with tf.device('/cpu:0'):

model_source = (
    './models/blanks/8L-64r2-128r2-256r2-512r2-K3-S2-P2-Arelu-D1024-D512')
model_dest = './models/8L-64r2-128r2-256r2-512r2-K3-S2-P2-Arelu-D1024-D512'

learning_rate = 0.001
BATCH_SIZE = 128

train_model(model_source, model_dest, BATCH_SIZE,
            learning_rate, from_json=True)
