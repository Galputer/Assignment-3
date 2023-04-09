from math import isqrt, sqrt
import numpy as np
import pandas as pd
from parse_board import parse_board

# Define a function to count the number of tiles in the correct position
def count_correct_tiles(board, goal):
    count = 0

    for i in range(len(board)):
        cur_tile = board[i]
        if goal[i] == 0: continue
        if cur_tile == goal[i]: count += 1
        
    return count

def sum_incorrect_weight(board, goal):
    sum = 0

    for i in range(len(board)):
        cur_tile = board[i]
        if cur_tile == 0 : continue
        if goal[i] == 0 and cur_tile != 0: sum += cur_tile
        elif cur_tile != goal[i]: sum += cur_tile
        
    return sum

def sort_board(board):
    sorted_board = np.sort([i for i in board if i != 0])
    return sorted_board

def goal_1(sorted_board, b_count) -> list: 
    sorted_board = [i for i in sorted_board]
    goal_list = sorted_board + [0]*b_count
    return goal_list

def goal_2(sorted_board, b_count) -> list:
    sorted_board = [i for i in sorted_board]
    goal_list = [0]*b_count + sorted_board
    return goal_list

def manhattan_distance(board, goal, n):
    """
    Calculates the Manhattan distance between the current board state and the goal state.
    """
    distance = 0
    for i in range(len(board)):
        if board[i] != 0 and goal[i] != 0 and board[i] != goal[i]:
            value = board[i]
            goal_pos = find_position(goal, value)
            distance += (abs(i % n - goal_pos % n) + abs(i // n - goal_pos // n))*value
    return distance

def find_position(goal_board, value):
    """
    Finds the index of a given value in the board.
    """
    for i in range(len(goal_board)):
        if goal_board[i] != 0 and goal_board[i] == value:
            return i

def linear_conflict(board, goal, n):
    """
    Calculates the linear conflict heuristic between the current board state and the goal state.
    """
    conflicts = 0
    for i in range(len(board)):
        if board[i] != 0 and board[i] != goal[i]:
            value = board[i]
            goal_pos = goal.index(value)
            if i // n == goal_pos // n:  # Same row
                for j in range(n * (i // n), n * (i // n + 1)):
                    other_tile = board[j]
                    if other_tile != 0 and goal[j] != 0 and other_tile != goal[j] and other_tile < value and goal.index(other_tile) // 3 == goal_pos // 3:
                        conflicts += value + other_tile
            if i % n == goal_pos % n:  # Same column
                for j in range(i % n, n*n, n):
                    other_tile = board[j]
                    if other_tile != 0 and goal[j] != 0 and other_tile != goal[j] and other_tile < value and goal.index(other_tile) % 3 == goal_pos % 3:
                        conflicts += value + other_tile
    return conflicts

def inversions_count(board):
    """
    Calculates the number of inversions in the given board state.
    """
    inversions = 0
    for i in range(len(board)):
        for j in range(i + 1, len(board)):
            if board[i] != 0 and board[j] != 0 and board[i] > board[j]:
                inversions += 1
    return inversions

def hamming_distance(board, goal):
    """
    Calculates the Hamming distance between two boards.
    """
    distance = 0
    for i in range(len(board)):
        if board[i] != goal[i]:
            distance += board[i]
    return distance

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
    df['inversion'] = df.apply(lambda x: inversions_count(x.board), axis=1)
    df['hamming_1'] = df.apply(lambda x: hamming_distance(x.board, x.goal_1), axis=1)
    df['hamming_2'] = df.apply(lambda x: hamming_distance(x.board, x.goal_2), axis=1)
    df['hamming'] = df.apply(lambda x: min(x.hamming_1, x.hamming_2), axis=1)
    
    # return df[['tile_sum','correct_1','correct_2','correct_count','incorrect_sum_1','incorrect_sum_2','incorrect_sum','manhattan_1','manhattan_2','manhattan','conflicts_1','conflicts_2','conflicts']]
    return df[['inversion','incorrect_sum','manhattan','conflicts','hamming']]

if __name__ == "__main__":

    board_df = parse_board()
    engineered_df = feature_engineer(board_df)
    print(engineered_df.loc[:10].to_string(index=False))
    print(engineered_df.shape[0])
    
    