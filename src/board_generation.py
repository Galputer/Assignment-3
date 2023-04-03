import csv
import random
import numpy as np
from src.npuzzle_helper import get_moved_board, valid_moves


def board_generation(n,b_count,board_count):
    """generate solvable boards

    Args:
        n (int): size n of the n*n board
        b_count (int): number of blanks in the board
        board_count (int): number of boards to generate

    Returns:
        save boards to ./boards
    """
    MOVES = 25
    for i in range(board_count):
        board_1d = [ele + 1 for ele in range(n**2)]
        for b in range(b_count):
            board_1d.pop(random.randrange(0,len(board_1d)))
        random_flag = random.randrange(0,2)
        if random_flag == 0:
            board_1d = ['B']*b_count + board_1d
        else:
            board_1d = board_1d + ['B']*b_count
        print('board_1d:',board_1d)
        board_2d = []
        for j in range(n):
            board_2d.append(board_1d[j*n:j*n+n])
        print('board_2d:',board_2d)
        
        for _ in range(MOVES):
            moves = valid_moves(board_2d)
            rand_move = moves[random.randrange(0,len(moves))]
            board_2d = get_moved_board(board_2d,rand_move)
        
        print('board_2d_moved:',board_2d)
        
        file_name = str(n)+"by"+str(n)+"_"+str(i)+".csv"

        np.savetxt('boards/'+file_name, 
            board_2d,
            delimiter =",", 
            fmt ='% s')
    return 0

if __name__ == "__main__":
    board_generation(4,2,5)