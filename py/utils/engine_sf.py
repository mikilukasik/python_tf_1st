import chess


class Engine_sf():
    def __init__(self, skill=2, depth=4):
        self.sf_engine = chess.engine.SimpleEngine.popen_uci(
            "/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish")
        self.skill = skill
        self.depth = depth

    def quit(self):
        self.sf_engine.quit()

    def get_move(self, board):
        self.sf_engine.configure({"Skill Level": self.skill})
        sf_move = self.sf_engine.play(
            board, chess.engine.Limit(depth=self.depth)).move

        return sf_move
