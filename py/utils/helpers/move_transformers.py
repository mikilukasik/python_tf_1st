def chess_move_to_indices(move):
    start_square = move.from_square
    end_square = move.to_square

    start_rank = 7 - (start_square // 8)
    start_file = chr(start_square % 8 + ord('a'))
    start_index = 8 * start_rank + (ord(start_file) - ord('a'))

    end_rank = 7 - (end_square // 8)
    end_file = chr(end_square % 8 + ord('a'))
    end_index = 8 * end_rank + (ord(end_file) - ord('a'))

    return (start_index, end_index)
