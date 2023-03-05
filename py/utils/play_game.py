import chess.engine
from helpers.board_with_lmft import get_board_with_lmft
import random

# Create a Stockfish engine instance
sf_engine = chess.engine.SimpleEngine.popen_uci(
    "/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish")

# Start a new game
board = get_board_with_lmft()

while not board.is_game_over():

    # Make a move
    sf_move = sf_engine.play(board, chess.engine.Limit(
        depth=3+random.uniform(-2, 2))).move
    print(sf_move)
    board.push(sf_move)

    # Print the board
    board.print()
    print()

    if board.is_game_over():
        break

    # Make a move
    sf_move = sf_engine.play(board, chess.engine.Limit(depth=3)).move
    print(sf_move)
    board.push(sf_move)

    # Print the board
    board.print()
    print()

print(board.result())


# Close the engine
sf_engine.quit()
