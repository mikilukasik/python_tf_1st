import chess.engine
from helpers.board_with_lmft import get_board_with_lmft
from helpers.engines import Engine
from utils.engine_sf import Engine_sf
import random
import time  # Import the time module
from utils.get_all_model_names import get_all_model_names

total_games = 200

# engine_a_name = 'inpConv_c16x2x5_skip_l2_d510_l1_bn/1.7275872230529785_best'
# engine_a_name = 'inpConv_c16x2x6_skip_l2_d510_l1_bn/1.7843259572982788_best'
# engine_a_name = 'inpConv_c16x2x6_skip_l2_d510_l1_bn/1.7889782670736314'
engine_a_name = 'inpConv_c16x2x6_skip_l2_d510_l1_bn/1.7753417491912842_best copy'
# engine_a_name = 'inpConv_c16x2x6_skip_l2_d510_l1_bn/1.7494691610336304_best'
# engine_a_name = 'inpConv_c16x2x6_skip_l2_d510_l1_bn/1.729777216911316_best copy'
# engine_a_name = 'inpConv_c16x2x6_skip_l2_d510_l1_bn/1.766247645020485 copy'
# engine_a_name = 'inpConv_c16x2x6_skip_l2_d510_l1_bn/1.8599706888198853_best copy'
# engine_a_name = 'inpConv_c16x2x5_skip_l2_d510_l1_bn/1.7275872230529785_best'
# engine_a_name = 'plain_c16x2x5_d101010_do1bc_v3/1.861648678779602_best'
# engine_a_name = 'currchampv1/1.786349892616272_best'
# engine_a_name = 'plain_c16x2x4_lc4d1_d0505_do1_bnorm_l2-4/4.364177227020264_best'

# engine_b_name = 'inpConv_c16x2x5_skip_l2_d510_l1_bn/1.7196457386016846_best'

engine_b_name = 'sf s1 d1'
engine_sf = Engine_sf(skill=1, depth=1)

engine_a = Engine(engine_a_name)
# engine_b = Engine(engine_b_name)
engine_b = engine_sf

wins_a = 0
wins_b = 0
draws = 0

total_time_a = 0.0
total_time_b = 0.0
moves_a = 0
moves_b = 0

for _ in range(total_games):
    board = get_board_with_lmft()

    # Determine which engine starts depending on the game number
    is_engine_a_turn = True if _ % 2 == 0 else False

    while not board.is_game_over():
        start_time = time.time()

        if is_engine_a_turn:
            move = engine_a.get_move(board)
            end_time = time.time()
            total_time_a += (end_time - start_time)
            moves_a += 1
        else:
            move = engine_b.get_move(board)
            end_time = time.time()
            total_time_b += (end_time - start_time)
            moves_b += 1

        board.push(move)

        # board.print()
        # print()

        # time.sleep(1)

        is_engine_a_turn = not is_engine_a_turn  # Switch turn

    result = board.result()

    if result == "1-0":
        if _ % 2 == 0:  # engine_a was white
            wins_a += 1
        else:  # engine_b was white
            wins_b += 1
    elif result == "0-1":
        if _ % 2 == 0:  # engine_a was black
            wins_b += 1
        else:  # engine_b was black
            wins_a += 1
    else:
        draws += 1

    print(f"Game {_+1}: {result}")

avg_time_a = total_time_a / moves_a
avg_time_b = total_time_b / moves_b

print(f"{engine_a_name} Wins: {wins_a}/{total_games}, {engine_b_name} Wins: {wins_b}/{total_games}, Draws: {draws}/{total_games}")
print(
    f"Average time taken by {engine_a_name} per move: {avg_time_a:.4f} seconds")
print(
    f"Average time taken by {engine_b_name} per move: {avg_time_b:.4f} seconds")

engine_sf.quit()
