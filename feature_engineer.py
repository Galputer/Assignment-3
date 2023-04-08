from math import isqrt, sqrt
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

def manhattan_distance(board, goal, n):
    """
    Calculates the Manhattan distance between the current board state and the goal state.
    """
    distance = 0
    for i in range(len(board)):
        if board[i] != "B" and goal[i] != "B" and board[i] != int(goal[i]):
            value = board[i]
            goal_pos = find_position(goal, value)
            distance += (abs(i % n - goal_pos % n) + abs(i // n - goal_pos // n))*value
    return distance

def find_position(goal_board, value):
    """
    Finds the index of a given value in the board.
    """
    for i in range(len(goal_board)):
        if goal_board[i] != 'B' and int(goal_board[i]) == value:
            return i

def linear_conflict(board, goal, n):
    """
    Calculates the linear conflict heuristic between the current board state and the goal state.
    """
    conflicts = 0
    for i in range(len(board)):
        if board[i] != "B" and board[i] != goal[i]:
            value = board[i]
            goal_pos = goal.index(str(value))
            if i // n == goal_pos // n:  # Same row
                for j in range(n * (i // n), n * (i // n + 1)):
                    other_tile = board[j]
                    if other_tile != "B" and goal[j] != "B" and other_tile != int(goal[j]) and other_tile < value and int(goal.index(str(other_tile))) // 3 == goal_pos // 3:
                        conflicts += value + other_tile
            if i % n == goal_pos % n:  # Same column
                for j in range(i % n, n*n, n):
                    other_tile = board[j]
                    if other_tile != "B" and goal[j] != "B" and other_tile != int(goal[j]) and other_tile < value and int(goal.index(str(other_tile))) % 3 == goal_pos % 3:
                        conflicts += value + other_tile
    return conflicts

def inversions_count_1(board):
    """
    Calculates the number of inversions in the given board state.
    """
    inversions = 0
    for i in range(len(board)):
        for j in range(i + 1, len(board)):
            if board[i] != "B" and board[j] != "B" and board[i] > board[j]:
                inversions += 1
    return inversions

def inversions_count_2(board):
    """
    Calculates the number of inversions in the given board state.
    """
    inversions = 0
    for i in range(len(board)):
        for j in range(i + 1, len(board)):
            if board[i] != "B" and board[j] != "B" and board[i] < board[j]:
                inversions += 1
    return inversions

def feature_engineer(df: pd.DataFrame):
    df['board_size'] = df['board'].apply(len)
    df['n'] = df['board_size'].apply(isqrt)
    df['sorted_board'] = df['board'].apply(sort_board)
    df['tile_sum'] = df['sorted_board'].apply(sum)
    df['goal_1'] = df.apply(lambda x: goal_1(x.sorted_board, x.b_count), axis=1)
    df['goal_2'] = df.apply(lambda x: goal_2(x.sorted_board, x.b_count), axis=1)
    df['correct_1'] = df.apply(lambda x: count_correct_tiles(x.board, x.goal_1), axis=1)
    df['correct_2'] = df.apply(lambda x: count_correct_tiles(x.board, x.goal_2), axis=1)
    df['correct_count'] = df.apply(lambda x: max(x.correct_1, x.correct_2), axis=1)
    df['incorrect_sum_1'] = df.apply(lambda x: sum_incorrect_weight(x.board, x.goal_1), axis=1)
    df['incorrect_sum_2'] = df.apply(lambda x: sum_incorrect_weight(x.board, x.goal_2), axis=1)
    df['incorrect_sum'] = df.apply(lambda x: min(x.incorrect_sum_1, x.incorrect_sum_1), axis=1)
    df['manhattan_1'] = df.apply(lambda x: manhattan_distance(x.board, x.goal_1, x.n), axis=1)
    df['manhattan_2'] = df.apply(lambda x: manhattan_distance(x.board, x.goal_2, x.n), axis=1)
    df['manhattan'] = df.apply(lambda x: min(x.manhattan_1, x.manhattan_2), axis=1)
    df['conflicts_1'] = df.apply(lambda x: linear_conflict(x.board, x.goal_1, x.n) + x.manhattan_1, axis=1)
    df['conflicts_2'] = df.apply(lambda x: linear_conflict(x.board, x.goal_2, x.n) + x.manhattan_2, axis=1)
    df['conflicts'] = df.apply(lambda x: min(x.conflicts_1, x.conflicts_2), axis=1)
    df['inversion_1'] = df.apply(lambda x: inversions_count_1(x.board), axis=1)
    df['inversion_2'] = df.apply(lambda x: inversions_count_2(x.board), axis=1)
    df['inversion'] = df.apply(lambda x: min(x.inversion_1, x.inversion_2), axis=1)
    
    
    # return df[['tile_sum','correct_1','correct_2','correct_count','incorrect_sum_1','incorrect_sum_2','incorrect_sum','manhattan_1','manhattan_2','manhattan','conflicts_1','conflicts_2','conflicts']]
    return df[['tile_sum','correct_count','inversion','incorrect_sum','manhattan','conflicts']]
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
    print(engineered_df.shape[0])
    
    