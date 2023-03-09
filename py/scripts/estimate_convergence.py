import os
import sys
from datetime import datetime

from utils.find_most_recent_model import find_most_recent_model
from utils.load_model import load_model_meta
from utils.plot_model_meta import plot_model_meta

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python find_metadata.py folder")
        sys.exit(1)

    folder = sys.argv[1]

    if not os.path.isdir(folder):
        print(folder, "is not a valid directory")
        sys.exit(1)

    modelFolder = find_most_recent_model(folder)

    if not modelFolder:
        print("No model.json files found in", folder)
        exit()

    model_meta = load_model_meta(modelFolder)

    # filename = f'./training_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    filename = f'./training_stats.pdf'
    print('filename', filename)

    plot_model_meta(model_meta, filename, True)
