import chess.engine
import numpy as np


def get_board_with_lmft():

    # create the vibrancy arrays, last_moved_from and last_moved_to
    lmf = np.zeros(10, dtype=np.uint8)
    lmt = np.zeros(10, dtype=np.uint8)

    # Start a new game
    board = chess.Board()
    push_move = board.push

    def push_move_new(*args, **kwargs):
        # print(f"pushing {args}, {kwargs}")
        push_move(*args, **kwargs)

    board.push = push_move_new

    def print_board(*args, **kwargs):
        print(board.unicode(empty_square="."))

    board.print = print_board

    return board
