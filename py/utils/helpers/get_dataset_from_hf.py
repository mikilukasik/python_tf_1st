import time
from datasets import load_dataset
from .board_with_lmft import get_board_with_lmft
from .get_xs import get_xs_without_zeros, get_ys_index, get_xs_new
from ..read_parquet_folder import read_parquet_folder
from collections import deque
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys

SOURCE_FOLDER = '/Volumes/Elements/dataset2/strategic_game_chess'

print('hello from get_dataset_from_hf.py')

board_states_set = set()
board_states_deque = deque(maxlen=1000000)
cache_20_percent = int(0.2 * 1000000)
cache_99_percent = int(0.99 * 1000000)
cache_randomized = False

skipped = 0


def game_to_csvish(line_from_hf):
    global skipped, cache_randomized, board_states_set, board_states_deque
    lines = []

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

        xs = get_xs_new(board)
        ys_index = get_ys_index(board, move)

        # xs_as_string = ','.join(str(x) for inner in xs for x in inner)
        # ys_index_as_string = str(ys_index)

        lines.append({'xs': xs, 'ys': ys_index})

        board.push_san(move)

    return lines


def fen_to_4bit_encoding(fen):
    piece_encoding = {
        'p': 0b0001, 'n': 0b0010, 'b': 0b0011, 'r': 0b0100, 'q': 0b0101, 'k': 0b0110,
        'P': 0b0111, 'N': 0b1000, 'B': 0b1001, 'R': 0b1010, 'Q': 0b1011, 'K': 0b1100,
        '1': 0b0000  # Representing an empty square
    }
    board_state = fen.split()[0]  # Extract the piece placement part of the FEN
    encoded_state = []

    for rank in board_state.split('/'):
        for char in rank:
            if char.isdigit():  # Repeat empty square encoding
                encoded_state.extend([piece_encoding['1']] * int(char))
            else:
                encoded_state.append(piece_encoding[char])

    # Convert the list of 4-bit values to a binary string or byte array
    # Assuming you want a byte array representation
    byte_array = bytearray()
    for i in range(0, len(encoded_state), 2):
        byte = (encoded_state[i] << 4) | encoded_state[i + 1]
        byte_array.append(byte)

    return byte_array


# # Example usage
# fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# encoded_state = fen_to_4bit_encoding(fen)


def game_to_moves(line_from_hf):
    global skipped, cache_randomized, board_states_set, board_states_deque
    lines = []  # List of lines as strings for the csv. comma separated xs values and then a single y index

    board = get_board_with_lmft()
    moves = line_from_hf['Moves']
    total_moves = len(moves)

    # Using enumerate to get move index
    for move_index, move in enumerate(moves):
        split_fen = board.fen().split(' ')
        board_state = split_fen[0]
        darks_turn = split_fen[1] == 'b'

        state = fen_to_4bit_encoding(board_state)
        lmf = board.lmf
        lmt = board.lmt
        ys = get_ys_index(board, move)

        if darks_turn:
            state, lmf, lmt, ys = flip(state, lmf, lmt, ys)

        lines.append((state, lmf, lmt,
                     ys, move_index, total_moves))

        board.push_san(move)

    return lines


def process_chunk(chunk):
    lines = []
    for line_from_hf in chunk:
        try:
            lines += game_to_csvish(line_from_hf)
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

        # self.iterable = read_parquet_folder(SOURCE_FOLDER)

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
                        # Break if no more games are available
                        break

                    # Process each batch in a separate thread
                    futures = [executor.submit(process_chunk, batch)
                               for batch in game_batches]
                    for future in as_completed(futures):
                        chunk_lines = future.result()
                        lines += chunk_lines

                        if len(lines) >= lines_to_get:
                            break  # Stop if enough lines are generated

                    print("Processed", self.counter, "games yo")
        # Trim to the requested number of lines
        # csv = '\n'.join(lines[:lines_to_get])
        print("Time to get", len(lines), "lines:",
              time.monotonic() - started, "s")
        print("skipped:", skipped)
        print("skip ratio:", skipped / (skipped + len(lines)))
        print("board states cache length:", len(board_states_deque))
        return lines
