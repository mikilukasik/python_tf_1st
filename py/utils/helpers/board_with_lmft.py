import chess.engine
import numpy as np
from .move_transformers import chess_move_to_indices


def add_one(array):
    return np.where(array < 255, array + 1, array)


def get_board_with_lmft():

    # Start a new game
    board = chess.Board()

    # create the vibrancy arrays, last_moved_from and last_moved_to
    board.lmf = np.full(64, 255, dtype=np.uint8)
    board.lmt = np.full(64, 255, dtype=np.uint8)

    push_move = board.push

    def push_move_new(*args, **kwargs):
        board.lmf = add_one(board.lmf)
        board.lmt = add_one(board.lmt)

        indices = chess_move_to_indices(args[0])

        board.lmf[indices[0]] = 1
        board.lmt[indices[1]] = 1

        push_move(*args, **kwargs)

    board.push = push_move_new

    def print_board(*args, **kwargs):
        print(board.unicode(empty_square="."))

    board.print = print_board

    return board
