from .predict_move import predict_move


class Engine():
    def __init__(self, model_name):
        # Store the model name as an instance variable
        self.model_name = model_name

    def get_move(self, board):  # Removed the 'engine' parameter
        move = predict_move(board, self.model_name)
        return move
