import time
from datasets import load_dataset
from .board_with_lmft import get_board_with_lmft
from .get_xs import get_xs_without_zeros, get_ys_index
from collections import deque
import threading
import random

import sys


board_states_set = set()
board_states_deque = deque(maxlen=1000000)
cache_20_percent = int(0.2 * 1000000)
cache_99_percent = int(0.99 * 1000000)
cache_randomized = False

skipped = 0


def game_to_csv(line_from_hf):
    global skipped, cache_randomized, board_states_set, board_states_deque
    # print(line_from_hf)
    # outputs {'Moves': ['d2d4', 'f7f5', 'g2g3', 'g7g6', 'f1g2', 'f8g7', 'g1f3', 'd7d6', 'c2c3', 'e7e6', 'a2a4', 'g8f6', 'd1c2', 'd8e7', 'b1d2', 'e6e5', 'd4e5', 'd6e5', 'e2e4', 'b8c6', 'e1g1', 'f5e4', 'd2e4', 'c8f5', 'f3d2', 'e8c8', 'b2b4', 'g7h6', 'f1e1', 'h6d2', 'c1d2', 'f6e4', 'g2e4', 'e7e6', 'd2g5', 'd8d6', 'a1d1', 'd6d1', 'e1d1', 'h7h6', 'g5e3', 'a7a5', 'c2b1', 'h6h5', 'b4b5', 'c6e7', 'e3g5', 'h8e8', 'h2h4', 'e6c4', 'd1e1', 'f5e4', 'e1e4', 'c4e6', 'g5f4', 'e6f5', 'f4e5', 'e7d5', 'b1e1', 'd5b6', 'f2f4', 'b6d7', 'e1e2', 'b7b6', 'e4e3', 'e8e7', 'e3e4', 'd7c5', 'e4d4', 'e7d7', 'g1g2', 'c8d8', 'g2h2', 'd8c8', 'e2g2', 'c8b8', 'g2a2', 'b8a7', 'a2g2', 'a7b8', 'g2e2', 'b8c8', 'e2f3', 'c8b8', 'f3d1', 'b8c8', 'd1e2', 'c8b8', 'e2d1', 'b8b7', 'd4d7', 'c5d7', 'e5d4', 'd7c5', 'h2g2', 'f5d5', 'g2g1', 'd5f5', 'd4c5', 'f5c5', 'd1d4', 'c5f5', 'd4d2', 'f5b1', 'g1f2', 'b1b3', 'd2d4', 'b3c2', 'f2e3', 'b7c8', 'd4h8', 'c8b7', 'h8d4', 'b7b8', 'd4d8', 'b8b7', 'd8d5', 'b7b8', 'd5g8', 'b8b7', 'g8c4', 'b7b8', 'c4g8', 'b8b7', 'g8d5', 'b7b8', 'd5d8', 'b8b7', 'd8d4', 'b7b8', 'd4d8', 'b8b7', 'd8d3', 'c2a4', 'd3g6', 'a4b5', 'g6e4', 'b7a7', 'f4f5', 'a5a4', 'f5f6', 'a4a3', 'f6f7', 'b5c5', 'e3e2', 'c5b5', 'e2e3', 'b5c5', 'e3d3', 'c5b5', 'd3e3', 'b5c5', 'e3d3', 'c5b5', 'e4c4', 'b5c4', 'd3c4', 'a3a2', 'f7f8q', 'a2a1q', 'f8f3', 'a1b1', 'f3h5', 'b1e4', 'c4b3', 'e4b1', 'b3a3', 'b1c1', 'a3b3', 'c1b1', 'b3c4', 'b1e4', 'c4b3', 'e4b1', 'b3a4', 'b1a2', 'a4b4', 'a2b1', 'b4a4', 'b1c2', 'a4b4', 'c2b1', 'b4a4', 'b1a2', 'a4b4', 'a2b1', 'b4a4', 'b1c2', 'a4b4', 'c2b1', 'b4c4', 'b1e4', 'c4b3', 'e4b1', 'b3c4', 'b1e4', 'c4b3', 'e4b1'], 'Termination': 'FIVEFOLD_REPETITION', 'Result': '1/2-1/2'}
    lines = []  # List of lines as strings for the csv. comma separated xs values and then a single y index

    board = get_board_with_lmft()
    for move in line_from_hf['Moves']:
        # check if board state (first part of fen) was served in the past million lines and if so, skip this line and push next move
        board_state = board.fen().split(' ')[0]
        cache_length = len(board_states_deque)
        if cache_length < cache_20_percent:
            skipped += 1
            if board_state not in board_states_set:
                board_states_set.add(board_state)
                board_states_deque.append(board_state)
            board.push_san(move)
            continue
        else:
            if board_state in board_states_set:
                board.push_san(move)
                skipped += 1
                continue
            else:
                if cache_length == 1000000:
                    # Remove the oldest state from the set if the deque is full
                    oldest_state = board_states_deque[0]
                    board_states_set.remove(oldest_state)
                board_states_set.add(board_state)
                board_states_deque.append(board_state)

                if cache_length >= cache_99_percent and not cache_randomized:
                    temp_list = list(board_states_deque)
                    random.shuffle(temp_list)
                    board_states_deque = deque(temp_list, maxlen=1000000)
                    print("randomized cache")
                    cache_randomized = True

        xs = get_xs_without_zeros(board)
        ys_index = get_ys_index(board, move)

        xs_as_string = ','.join(str(x) for inner in xs for x in inner)
        ys_index_as_string = str(ys_index)

        lines.append(xs_as_string + ',' + ys_index_as_string)

        board.push_san(move)

    return lines


class ChessDataset:
    _lock = threading.Lock()

    def __init__(self):
        # Load dataset and create iterable upon initialization
        started = time.monotonic()
        self.dataset = load_dataset(
            "laion/strategic_game_chess", 'en', streaming=True)["train"]
        self.iterable = iter(self.dataset)
        print("Time to load dataset:", time.monotonic() - started, "s")

    def get_dataset_as_csv(self, lines_to_get):
        global skipped
        skipped = 0
        # Yield the requested number of lines
        started = time.monotonic()
        lines = []

        with ChessDataset._lock:  # Acquire the lock
            while len(lines) < lines_to_get:
                try:
                    lines += game_to_csv(next(self.iterable))
                except StopIteration:
                    print("Reached end of dataset, restarting iterable")
                    self.iterable = iter(self.dataset)

        csv = '\n'.join(lines)
        print("Time to get", len(lines), "lines:",
              time.monotonic() - started, "s")
        print("skipped:", skipped)
        print("skip ratio:", skipped / (skipped + len(lines)))

        print("board states cache length:", len(board_states_deque))
        return csv


# # Example usage
# chess_dataset = ChessDataset()
# for line in chess_dataset.get_dataset(25000):
#     print(line)
