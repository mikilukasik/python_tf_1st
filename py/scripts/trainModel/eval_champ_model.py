from utils.train_model_v6 import train_model
from utils.find_most_recent_model import find_most_recent_model
import os
import sys
from datetime import datetime

from utils.find_most_recent_model import find_most_recent_model
from utils.load_model import load_model_meta
from utils.plot_model_meta import plot_model_meta

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_champ_model.py folder")
        sys.exit(1)

    folder = sys.argv[1]

    if not os.path.isdir(folder):
        print(folder, "is not a valid directory")
        sys.exit(1)

    model_folder = find_most_recent_model(folder)

    if not model_folder:
        print("No model.json files found in", folder)
        exit()

    model_meta = load_model_meta(model_folder)

    model_dest = '../models/dummy' 

    initial_batch_size = 160
    lr_multiplier = 1
    fixed_lr = 0.00003
    # fixed_lr = 0.00000000003

    train_model(model_folder, model_dest,
                initial_batch_size, gpu=True, lr_multiplier=lr_multiplier, fixed_lr=fixed_lr, dataset_reader_version='18', filter='2700', ys_format='default', evaluateOnly=True)
