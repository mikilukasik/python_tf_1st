# from utils.train_model import train_model

# fmt: off
import sys
sys.path.append('./py/utils')
from train_model import train_model
# fmt: on

# with tf.device('/cpu:0'):

model_source = (
    './models/blanks/CL4-F64-64-128-128-R2-DL2-DU1024-512-Arelu-GBN0-GAP0-Att0')
model_dest = './models/CL4-F64-64-128-128-R2-DL2-DU1024-512-Arelu-GBN0-GAP0-Att0'

learning_rate = 0.0001
BATCH_SIZE = 128

train_model(model_source, model_dest, BATCH_SIZE,
            learning_rate, from_json=True)
