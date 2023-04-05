import numpy as np
import pandas as pd
from parse_board import parse_board

# Define a function to count the number of tiles in the correct position
def count_correct_tiles(board, goal):
    count = 0

    for i in range(len(board)):
        cur_tile = board[i]
        if goal[i] == 'B': continue
        if cur_tile == int(goal[i]): count += 1
        
    return count

def sum_incorrect_weight(board, goal):
    sum = 0

    for i in range(len(board)):
        cur_tile = board[i]
        if cur_tile == 'B' : continue
        if goal[i] == 'B' and cur_tile != 'B': sum += cur_tile
        elif cur_tile != int(goal[i]): sum += cur_tile
        
    return sum

def sort_board(board):
    sorted_board = np.sort([i for i in board if i != 'B'])
    return sorted_board

def goal_1(sorted_board, b_count) -> list: 
    sorted_board = [str(i) for i in sorted_board]
    goal_list = sorted_board + ["B"]*b_count
    return goal_list

def goal_2(sorted_board, b_count) -> list:
    sorted_board = [str(i) for i in sorted_board]
    goal_list = ["B"]*b_count + sorted_board
    return goal_list

def feature_engineer(df: pd.DataFrame):
    df['board_size'] = df['board'].apply(len)
    df['sorted_board'] = df['board'].apply(sort_board)
    df['tile_sum'] = df['sorted_board'].apply(sum)
    df['goal_1'] = df.apply(lambda x: goal_1(x.sorted_board, x.b_count), axis=1)
    df['goal_2'] = df.apply(lambda x: goal_2(x.sorted_board, x.b_count), axis=1)
    df['correct_1'] = df.apply(lambda x: count_correct_tiles(x.board, x.goal_1), axis=1)
    df['correct_2'] = df.apply(lambda x: count_correct_tiles(x.board, x.goal_2), axis=1)
    df['incorrect_sum_1'] = df.apply(lambda x: sum_incorrect_weight(x.board, x.goal_1), axis=1)
    df['incorrect_sum_2'] = df.apply(lambda x: sum_incorrect_weight(x.board, x.goal_2), axis=1)
    
    return df[['b_count', 'board_size', 'tile_sum','correct_1','correct_2','incorrect_sum_1','incorrect_sum_2']]
if __name__ == "__main__":
    # board_1d = np.array(['B', 'B', 5, 3, 10, 6, 11, 7, 4, 8, 9, 16, 12, 'B', 13, 15])
    # board_sorted = [3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 15, 16]
    # goal_lst = ["B","B","B",3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 15, 16]
    # print(count_correct_tiles(board_1d,goal_lst))
    # print(sum_incorrect_weight(board_1d,goal_lst))
    # b_count = 3
    # print(goal_2(board_sorted,b_count))
    # print(num_adjacent_to_blank(board_1d))
    board_df = parse_board()
    engineered_df = feature_engineer(board_df)
    print(engineered_df.loc[:10].to_string(index=False))
    
    