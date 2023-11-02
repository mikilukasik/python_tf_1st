import chess.engine
from helpers.board_with_lmft import get_board_with_lmft
from helpers.engines import Engine
from utils.engine_sf import Engine_sf
import time
import os
from datetime import datetime


# Generate the current timestamp and format it as YYYYMMDD_HHMMSS
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

stockfish_games = 250

# Constants for Chess Results
WHITE_WIN = "1-0"
BLACK_WIN = "0-1"
DRAW = "1/2-1/2"

engine_names = [
    'inpConv_c16x2x6_skip_l2_d101010_l1_do1_bn/1.7162982078790665 copy',
    'inpConv_c16x2x6_skip_l2_d101010_l1_do1_bn/1.7314529418945312_best copy',
    'inpConv_c16x2x6_skip_l2_d101010_l1_do1_bn/1.749123655430814_temp',
    'inpConv_c16x2x6_skip_l2_d101010_l1_do1_bn/1.7490653453380365_temp',
    'inpConv_c16x2x6_skip_l2_d510_l1_bn/1.7753417491912842_best copy',
    'inpConv_c16x2x5_skip_l2_d510_l1_bn/1.7275872230529785_best',
    'inpConv_c16x2x6_skip_l2_d510_l1_bn/1.7226097583770752_best copy',
    'inpConv_c16x2x6_skip_l2_d510_l1_bn/1.7330505576133728 copy',
    'inpConv_c16x2x6_skip_l2_d510_l1_bn/1.7388049893379212 copy',
    'inpConv_c16x2x6_skip_l2_d510_l1_bn/1.7381287091970443_temp',
    'inpConv_c16x2x6_skip_l2_d510_l1_bn/1.729777216911316_best copy',
    'inpConv_c16x2x6_skip_l2_d510_l1_bn/1.766247645020485 copy',
    'inpConv_c16x2x6_skip_l2_d510_l1_bn/1.8599706888198853_best copy',
    'plain_c16x2x5_d101010_do1bc_v3/1.861648678779602_best',
    'currchampv1/1.786349892616272_best',
    'plain_c16x2x4_lc4d1_d0505_do1_bnorm_l2-4/4.364177227020264_best'
]


def find_latest_game_result_file():
    """Find the latest game results file based on the timestamp."""
    files = [f for f in os.listdir('.') if f.startswith('game_results_')]
    return max(files, key=os.path.getctime) if files else None


class Tournament:

    def __init__(self):
        self.results = {}
        # Load previously played games
        latest_file = find_latest_game_result_file()

        print(f"Latest file: {latest_file}")

        self.previous_results = {}
        if latest_file:
            with open(latest_file, 'r') as f:
                for line in f:
                    matchup, result = line.strip().split(': ')
                    self.previous_results[matchup] = result

    def play_game(self, engine_white, engine_black):
        """Play a single game between two engines and return the result."""
        matchup_name = self._get_matchup_name(engine_white, engine_black)
        # Use the cached result if found for non-stockfish games
        if matchup_name in self.previous_results and "Stockfish" not in matchup_name:
            return self.previous_results[matchup_name]

        board = get_board_with_lmft()
        while not board.is_game_over():
            move = engine_white.get_move(
                board) if board.turn == chess.WHITE else engine_black.get_move(board)
            board.push(move)
        return board.result()

    def play_tournament(self, engines):
        """Play a round-robin tournament among provided engines."""
        for i in range(len(engines)):
            for j in range(len(engines)):
                if i != j:
                    white = engines[i]
                    black = engines[j]
                    result = self.play_game(white, black)
                    matchup_name = self._get_matchup_name(white, black)

                    # Append result to the game_results.txt
                    filename = f"game_results_{timestamp}.txt"
                    with open(filename, "a") as f:
                        f.write(f"{matchup_name}: {result}\n")

                    self.results[matchup_name] = result
                    self._log_current_ranking()

    def _get_matchup_name(self, white, black):
        """Generate a matchup name for two engines."""
        return f"{white.name} vs {black.name}"

    def _log_current_ranking(self):
        """Log the current ranking based on the results."""
        points = self._calculate_points()
        games_played = self._calculate_games_played()
        sorted_engines = sorted(points, key=lambda x: (
            points[x] / games_played[x], points[x]), reverse=True)
        print("\nCurrent Ranking:")
        for engine in sorted_engines:
            print(
                f"{engine}: {points[engine]} from {games_played[engine]} games ({points[engine] / games_played[engine]:.2f} ppg)")
        print("\n")

        # Update the current ranking file
        filename = f"current_rankings_{timestamp}.txt"
        with open(filename, "w") as f:
            for engine in sorted_engines:
                f.write(
                    f"{engine}: {points[engine]} from {games_played[engine]} games ({points[engine] / games_played[engine]:.2f} ppg)\n")

    def _calculate_points(self):
        """Calculate points for each engine based on game results."""
        points = {}
        for k, v in self.results.items():
            white, black = k.split(" vs ")
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
            white, black = k.split(" vs ")
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

    def count_stockfish_games(self, engine_name):
        """Count the number of games an engine has played against Stockfish."""
        count = 0
        for matchup in self.previous_results.keys():
            if "Stockfish" in matchup and engine_name in matchup:
                count += 1
        return count

    def play_stockfish_games(self, engines, engine_sf):
        """Play each engine against Stockfish x times."""
        for engine in engines:
            games_played = self.count_stockfish_games(engine.name)
            games_to_play = stockfish_games - games_played

            for idx in range(games_to_play):
                if idx % 2 == 0:
                    result = self.play_game(engine, engine_sf)
                    key = f"{engine.name} vs Stockfish"
                    individual_key = f"{engine.name} vs Stockfish Game {idx + games_played + 1}"
                else:
                    result = self.play_game(engine_sf, engine)
                    key = f"Stockfish vs {engine.name}"
                    individual_key = f"Stockfish vs {engine.name} Game {idx + games_played + 1}"

                # Append individual game result to the game_results.txt
                with open(f"game_results_{timestamp}.txt", "a") as f:
                    f.write(f"{individual_key}: {result}\n")

                # Aggregate results for ranking
                if key in self.results:
                    if result == WHITE_WIN:
                        self.results[key][0] += 1
                    elif result == BLACK_WIN:
                        self.results[key][1] += 1
                    else:
                        # Increment the number of draws by 1
                        self.results[key][2] += 1
                else:
                    if result == WHITE_WIN:
                        self.results[key] = [1, 0, 0]
                    elif result == BLACK_WIN:
                        self.results[key] = [0, 1, 0]
                    else:
                        # Initialize with one draw
                        self.results[key] = [0, 0, 1]

                self._log_current_ranking()


if __name__ == "__main__":

    # Initialize tournament and engines
    tournament = Tournament()
    engines = [Engine(name) for name in engine_names]
    engine_sf = Engine_sf(skill=1, depth=1, name="Stockfish")

    # Play Engines against each other
    tournament.play_tournament(engines)

    # Play Engines against Stockfish many times each
    tournament.play_stockfish_games(engines, engine_sf)

    engine_sf.quit()
