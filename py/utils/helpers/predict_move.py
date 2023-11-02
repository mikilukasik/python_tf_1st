from utils.load_model import load_model
import os
import threading
import numpy as np
import tensorflow as tf
import chess
import json

# Specify the path to your JSON file
file_path = "./utils/helpers/movesToOneHotMapV2.json"

# Load the JSON data
with open(file_path, 'r') as file:
    move_map = json.load(file)


def mirrorCell(cellIndex):
    rank = cellIndex >> 3
    file = cellIndex & 7
    return ((7 - rank) << 3) + file


def create_inverted_map(move_map):
    inverted_map = {}
    for from_square, targets in move_map.items():
        for to_square, promotions in targets.items():
            for promotion, ys_index in promotions.items():
                inverted_from_square = str(mirrorCell(int(from_square)))
                inverted_to_square = str(mirrorCell(int(to_square)))

                if inverted_from_square not in inverted_map:
                    inverted_map[inverted_from_square] = {}

                if inverted_to_square not in inverted_map[inverted_from_square]:
                    inverted_map[inverted_from_square][inverted_to_square] = {}

                inverted_map[inverted_from_square][inverted_to_square][promotion] = ys_index
    return inverted_map


inverted_move_map = create_inverted_map(move_map)


models_folder = '../models'
model_cache = {}
model_timers = {}

promotionPieces = ["", None, "b", "n", "r", "q"]


def clear_model_cache(model_name):
    # print('Cleaning up after model:', model_name)
    model_cache.pop(model_name, None)
    model_timers.pop(model_name, None)


def cellIndex2cellStr(index):
    return f"{chr((index % 8) + 97)}{8 - (index // 8)}"


def getPromotionPieceFromMove(move):
    newPiece = (move >> 6) & 7
    return promotionPieces[newPiece]


def move2moveString(move):
    return f"{cellIndex2cellStr(move >> 10)}{cellIndex2cellStr(move & 63)}{getPromotionPieceFromMove(move)}"


def mirrorMove(move):
    sourceIndex = move >> 10
    targetIndex = move & 63
    piece = (move >> 6) & 15
    newPiece = piece ^ 8 if piece else 0
    return (mirrorCell(sourceIndex) << 10) + mirrorCell(targetIndex) + (newPiece << 6)


def getLmVal(allValsArr, index):
    return 1 / allValsArr[index]


def mirrorFlatArray(arr):
    chunks = [arr[i:i+8] for i in range(0, len(arr), 8)][::-1]
    return [item for sublist in chunks for item in sublist]


def invert_case(char):
    if char.isalpha():
        return char.lower() if char.isupper() else char.upper()
    return char


def mirrorBoard(arr):
    chunks = [arr[i:i+8] for i in range(0, len(arr), 8)][::-1]

    # Invert the case of characters and flatten the chunks back into a single list
    return [invert_case(item) for sublist in chunks for item in sublist]


def expand_fen_board_state(board_state):
    expanded_state = []

    for char in board_state:
        if char.isdigit():
            expanded_state.extend(['1'] * int(char))
        elif char != '/':  # Exclude slashes
            expanded_state.append(char)

    return (expanded_state)


def extract_board_and_turn(board):
    fen_parts = board.fen().split()
    board_state = expand_fen_board_state(fen_parts[0])
    is_white_to_move = fen_parts[1] == "w"
    return board_state, is_white_to_move


piece_map = {
    'p': 0, 'b': 1, 'n': 2, 'r': 3, 'q': 4, 'k': 5,
    'P': 6, 'B': 7, 'N': 8, 'R': 9, 'Q': 10, 'K': 11
}


def getXs(board):
    xsAsArray = []

    origLmt = board.lmt
    origLmf = board.lmf

    board_state, is_white_next = extract_board_and_turn(board)

    if is_white_next:
        board = board_state
        lmf = origLmf
        lmt = origLmt
    else:
        board = mirrorBoard(board_state)
        lmf = mirrorFlatArray(origLmf)
        lmt = mirrorFlatArray(origLmt)

    arr = []
    for piece in board:

        one_hot_encoding = [0] * 12

        if piece in piece_map:
            one_hot_encoding[piece_map[piece]] = 1
            arr.append(one_hot_encoding)
        else:
            for _ in range(int(piece)):
                arr.append([0] * 12)

    xsAsArray.extend([[*arr[i], 1/(lmf[i]), 1/(lmt[i])]
                     for i in range(64)])

    return xsAsArray


def ysToWinningMove(moveYs, board):

    current_map = inverted_move_map if board.turn else move_map

    # Given the map format, get the ys index for a move

    def get_ys_index(move):
        from_square = str(move.from_square)
        to_square = str(move.to_square)

        # If the move involves a promotion, consider it, else default to ""
        promotion = move.promotion if move.promotion in [
            chess.QUEEN, chess.KNIGHT] else ""

        if promotion == chess.QUEEN:
            promotion = ""
        elif promotion == chess.KNIGHT:
            promotion = "n"

        # If the move exists in the map, return its corresponding ys index
        if from_square in current_map and to_square in current_map[from_square] and promotion in current_map[from_square][to_square]:
            return current_map[from_square][to_square][promotion]
        return None

    # For each legal move, find the move with the highest ys value
    best_move = None
    best_value = float('-inf')
    for move in board.legal_moves:
        index = get_ys_index(move)
        if index is not None and moveYs[index] > best_value:
            best_value = moveYs[index]
            best_move = move

    return best_move


def predict_move(board, model_name):

    # Load the model from the cache or from disk
    model = model_cache.get(model_name)
    if model is None:
        model_path = os.path.join(
            models_folder, model_name)
        # print('Loading model:', model_name)
        model = load_model(model_path, quiet=True)
        model_cache[model_name] = model
        # print('Loaded model:', model_name)

    xsAsArray = getXs(board)
    boardXs = tf.convert_to_tensor(xsAsArray, dtype=tf.float32)
    boardXs = tf.reshape(boardXs, [1, 8, 8, 14])
    moveYs = model.predict(boardXs, verbose=0)

    # Set a timer to clear the model from the cache after half a minute
    if model_name in model_timers:
        model_timers[model_name].cancel()
    timer = threading.Timer(30, clear_model_cache, args=[model_name])
    model_timers[model_name] = timer
    timer.start()

    return ysToWinningMove(moveYs[0], board)
