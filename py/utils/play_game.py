import chess.engine
from helpers.board_with_lmft import get_board_with_lmft
from helpers.engines import Engine
from utils.engine_sf import Engine_sf
import time
import os
from datetime import datetime


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

stockfish_games = 1000

stockfish_depth = 5
stockfish_skill = 5

stockfish_name = "Stockfish d" + \
    str(stockfish_depth) + " s" + str(stockfish_skill)

WHITE_WIN = "1-0"
BLACK_WIN = "0-1"
DRAW = "1/2-1/2"

engine_names = [
    # "merged_double/v1/1.5734939713162535 copy",
    'merged_double/_orig',
    # 'merged_triple/_orig',
    # 'merged_triple_v2/_orig',
    'merged_2op3me/_orig',
    # 'merged_XL3_train_midend/1.6300672443740625 copy',
    # 'merged_XL3_train_midend/1.6535537508027307 copy',
    'merged_XL3/_orig',
    'merged_XL/_orig',
    # '5+1fold_XS/1.8239935278892516_temp',
    'merged_3models_fo/1.590137270045908 copy',
    'merged_mg4/1.586208572149277_temp',

    'merged_mg3/1.5581268072128296_best copy',
    #: 675.5 from 1048 games (0.64 ppg)

    'merged_mg4/1.5850087277119673_temp',
    #: 660.5 from 1048 games (0.63 ppg)

    'merged_trained/1.6335621824351754 copy',
    #: 655.0 from 1048 games (0.62 ppg)

    'merged_mg3/1.5735136878617266 copy',
    #: 649.0 from 1048 games (0.62 ppg)

    'merged_trained_progFixed/1.6200499534606934_best copy',
    #: 648.5 from 1048 games (0.62 ppg)

    'merged_mg4/1.582800602197647 copy',
    #: 646.0 from 1048 games (0.62 ppg)

    'merged_trained/1.631397050023079 copy',
    #: 644.0 from 1048 games (0.61 ppg)

    'merged_mg2/1.568634271621704_best copy',
    #: 643.5 from 1048 games (0.61 ppg)

    'merged_trained_progFixed/1.596966028213501_best copy',
    #: 641.5 from 1048 games (0.61 ppg)

    'merged_mg4/1.5783308744430542_best copy',
    #: 640.5 from 1048 games (0.61 ppg)

    'merged_trained/1.629942218542099 copy',
    #: 638.5 from 1048 games (0.61 ppg)

    'merged_trained_progFixed/_orig',
    #: 638.0 from 1048 games (0.61 ppg)

    'merged_trained_progFixed/1.6103842393159866 copy',
    #: 637.5 from 1048 games (0.61 ppg)

    'merged_mg3/1.5500810146331787_best copy',
    #: 637.0 from 1048 games (0.61 ppg)

    'inpConv_c16x2x6_skip_l2_d101010_l1_do1_bn/1.6967400099549974',
]


def find_latest_game_result_files(n=30):
    """Find the latest 'n' game results files based on the timestamp."""
    files = [f for f in os.listdir('.') if f.startswith('game_results_')]
    sorted_files = sorted(files, key=os.path.getctime, reverse=True)
    return sorted_files[:n] if files else []


class Tournament:
    def __init__(self):
        self.results = {}
        self.buffer = []
        self.last_write_time = time.time()

        # Load previously played games from the last 15 files
        latest_files = find_latest_game_result_files(30)

        print(f"Latest 15 files: {latest_files}")

        self.previous_results = {}
        for file in latest_files:
            with open(file, 'r') as f:
                for line in f:
                    matchup, result = line.strip().split(': ')
                    # Update only if the matchup is not already in the results
                    if matchup not in self.previous_results:
                        self.previous_results[matchup] = result

    def play_game(self, engine_white, engine_black, matchup_name=None):
        """Play a single game between two engines and return the result."""
        if matchup_name is None:
            matchup_name = self._get_matchup_name(engine_white, engine_black)
        # Use the cached result if found for non-stockfish games
        if matchup_name in self.previous_results:
            return self.previous_results[matchup_name]

        board = get_board_with_lmft()
        while not board.is_game_over():
            move = engine_white.get_move(
                board) if board.turn == chess.WHITE else engine_black.get_move(board)
            board.push(move)

        self.buffer.append(f"{matchup_name}: {board.result()}\n")
        self.maybe_write_buffer_to_file()

        return board.result()

    def play_tournament(self, engines):
        """Play a round-robin tournament among provided engines."""
        for i in range(len(engines)):
            for j in range(len(engines)):
                if i != j:
                    white = engines[i]
                    black = engines[j]

                    for k in range(stockfish_games if white.name == stockfish_name or black.name == stockfish_name else 1):

                        matchup_name = self._get_matchup_name(
                            white, black) + (f" Game {k + 1}" if k > 0 else '')
                        result = self.play_game(white, black, matchup_name)

                        self.results[matchup_name] = result
                        self._log_current_ranking()

        self.write_buffer_to_file()

    def _get_matchup_name(self, white, black):
        """Generate a matchup name for two engines."""
        return f"{white.name} vs {black.name}"

    def write_buffer_to_file(self):
        """Write the buffered results to the file."""
        filename = f"game_results_{timestamp}.txt"
        with open(filename, "a") as f:
            for line in self.buffer:
                f.write(line)
        self.buffer.clear()
        self.last_write_time = time.time()

    def maybe_write_buffer_to_file(self):
        """Check if it's time to write the buffer to the file."""
        if time.time() - self.last_write_time >= 1:
            self.write_buffer_to_file()

    def _log_current_ranking(self):
        """Log the current ranking based on the results."""

        # Ensure we write buffered results before logging current ranking
        self.maybe_write_buffer_to_file()

        points = self._calculate_points()
        games_played = self._calculate_games_played()
        sorted_engines = sorted(points, key=lambda x: (
            points[x] / games_played[x], points[x]), reverse=True)
        filename = f"current_rankings_{timestamp}.txt"
        with open(filename, "w") as f:
            for engine in sorted_engines:
                f.write(
                    f"{engine}: {points[engine]} from {games_played[engine]} games ({points[engine] / games_played[engine]:.2f} ppg)\n")

    def _remove_game_suffix(self, matchup):
        """Remove the ' Game n' suffix from the matchup name, if present."""
        import re
        return re.sub(r' Game \d+$', '', matchup)

    def _calculate_points(self):
        """Calculate points for each engine based on game results."""
        points = {}
        for k, v in self.results.items():
            white, black = self._remove_game_suffix(k).split(" vs ")
            if isinstance(v, list):  # This is a Stockfish matchup
                points[white] = points.get(white, 0) + v[0] + 0.5 * v[2]
                points[black] = points.get(black, 0) + v[1] + 0.5 * v[2]
            else:
                if v == WHITE_WIN:
                    points[white] = points.get(white, 0) + 1
                elif v == BLACK_WIN:
                    points[black] = points.get(black, 0) + 1
                else:  # Draw
                    points[white] = points.get(white, 0) + 0.5
                    points[black] = points.get(black, 0) + 0.5
        return points

    def _calculate_games_played(self):
        """Calculate the number of games played by each engine."""
        games_played = {}
        for k, v in self.results.items():
            white, black = self._remove_game_suffix(k).split(" vs ")
            if isinstance(v, list):  # Stockfish matchup
                total_games_for_matchup = sum(v)
                games_played[white] = games_played.get(
                    white, 0) + total_games_for_matchup
                games_played[black] = games_played.get(
                    black, 0) + total_games_for_matchup
            else:
                games_played[white] = games_played.get(white, 0) + 1
                games_played[black] = games_played.get(black, 0) + 1
        return games_played


if __name__ == "__main__":
    tournament = Tournament()
    engine_sf = Engine_sf(name=stockfish_name, skill=stockfish_skill,
                          depth=stockfish_depth)
    engines = [Engine(name) for name in engine_names] + [engine_sf]

    tournament.play_tournament(engines)
    engine_sf.quit()
