import os
import sys
import logging

from .predict_move import predict_move

logging.basicConfig(level=logging.ERROR)  # Setup basic logging configuration
models_folder = '../models'


class Engine():
    def __init__(self, model_name):
        # Store the model name as an instance variable
        self.model_name = model_name

        # Check if the folder exists
        model_path = os.path.join(
            models_folder, model_name)

        if not os.path.exists(model_path) or not os.path.isdir(model_path):
            logging.error(f"Model folder '{self.model_name}' does not exist.")
            sys.exit(1)

    @property
    def name(self):
        return self.model_name

    def get_move(self, board):  # Removed the 'engine' parameter
        move = predict_move(board, self.model_name)
        return move
