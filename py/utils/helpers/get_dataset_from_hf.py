import time
from datasets import load_dataset
from .board_with_lmft import get_board_with_lmft
from .get_xs import get_xs_without_zeros, get_ys_index
from collections import deque
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys

print('hello from get_dataset_from_hf.py')

board_states_set = set()
board_states_deque = deque(maxlen=1000000)
cache_20_percent = int(0.2 * 1000000)
cache_99_percent = int(0.99 * 1000000)
cache_randomized = False

skipped = 0


def game_to_csv(line_from_hf):
    global skipped, cache_randomized, board_states_set, board_states_deque
    lines = []  # List of lines as strings for the csv. comma separated xs values and then a single y index

    board = get_board_with_lmft()
    for move in line_from_hf['Moves']:
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


def process_chunk(chunk):
    global board_states_set, board_states_deque
    lines = []
    for line_from_hf in chunk:
        try:
            lines += game_to_csv(line_from_hf)
        except StopIteration:
            continue  # Ignore StopIteration and continue with the next line
    return lines


class ChessDataset:
    _lock = threading.Lock()

    def __init__(self):
        # Load dataset and create iterable upon initialization
        started = time.monotonic()
        self.dataset = load_dataset(
            "laion/strategic_game_chess", 'en', streaming=True)["train"]
        self.iterable = iter(self.dataset)
        self.counter = 0
        print("Time to load dataset:", time.monotonic() - started, "s")

    def get_dataset_as_csv(self, lines_to_get, num_threads=8):
        global skipped
        print("Getting", lines_to_get, "lines from dataset")
        skipped = 0
        started = time.monotonic()
        lines = []

        with ChessDataset._lock:
            while len(lines) < lines_to_get:
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    # Fetch batches of games for processing (30 games per thread)
                    game_batches = []
                    for _ in range(num_threads):
                        batch = [next(self.iterable, None) for _ in range(30)]
                        batch = [game for game in batch if game is not None]
                        self.counter += len(batch)
                        if batch:
                            game_batches.append(batch)

                    if not game_batches:
                        print("Reached end of dataset, restarting iterable")
                        # Reset the dataset iterator
                        self.iterable = iter(self.dataset)
                        continue

                    # Process each batch in a separate thread
                    futures = [executor.submit(process_chunk, batch)
                               for batch in game_batches]
                    for future in as_completed(futures):
                        chunk_lines = future.result()
                        lines += chunk_lines

                        if len(lines) >= lines_to_get:
                            break  # Stop if enough lines are generated

                    print("Processed", self.counter, "games")
        # Trim to the requested number of lines
        csv = '\n'.join(lines[:lines_to_get])
        print("Time to get", len(lines), "lines:",
              time.monotonic() - started, "s")
        print("skipped:", skipped)
        print("skip ratio:", skipped / (skipped + len(lines)))
        print("board states cache length:", len(board_states_deque))
        return csv
