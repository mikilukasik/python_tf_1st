import chess.engine
from helpers.board_with_lmft import get_board_with_lmft
from helpers.engines import Engine
import random

# Create a Stockfish engine instance
sf_engine = chess.engine.SimpleEngine.popen_uci(
    "/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish")

my_engine = Engine()

# Start a new game
board = get_board_with_lmft()

while not board.is_game_over():

    # Make a move
    sf_move = sf_engine.play(board, chess.engine.Limit(
        depth=3+random.uniform(-2, 2))).move
    print('sf move:', sf_move)
    board.push(sf_move)

    # Print the board
    board.print()
    print()

    if board.is_game_over():
        break

    # Make a move
    my_move = my_engine.get_move(board)
    print('my engine move:', my_move)
    board.push(my_move)

    # Print the board
    board.print()
    print()

print(board.result())


# Close the engine
sf_engine.quit()
