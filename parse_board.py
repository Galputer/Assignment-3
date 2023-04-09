import csv
import os
import numpy as np
import pandas as pd

def parse_board():
    directory = 'board-data'
    board_df = pd.DataFrame(columns=["board","astar","b_count"])

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and 'csv' in filename:
            board_data,a_star,b_count = read_file(f)
            board_df.loc[len(board_df)] = [board_data, a_star, b_count]
    
    agg_df = read_aggregated()
    return pd.concat([board_df, agg_df], axis=0,ignore_index=True)
    
def read_file(file_path):
    """Read the board file from csv to memory.
    Return the board content and number of B letters.
    Args:
        file_path (String): The path string for input board csv file
    Returns:
        board_data: 1D array of board file
        a_star: astar value of the board
    """
    datafile = open(file_path, 'r', encoding="utf-8-sig")
    datareader = csv.reader(datafile, delimiter=',')
    data = []
    B_count = 0
    for row in datareader:
        for element in row:
            if element.isnumeric():
                data.append(int(element))
            else:
                data.append(0)
                B_count += 1
    board_data = data[:-1]
    a_star = data[-1]
    return board_data,a_star,B_count

def count_B(arr):
    count = 0
    for element in arr:
        if element == 0: count += 1
    return count

def read_aggregated():
    directory = 'aggregated-board-data'
    lst_board_filename = ['3x3_1BLANK_10000-1.csv','3x3_2BLANK_10000.csv']
    lst_board = []
    for filename in lst_board_filename:
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and 'csv' in filename:
            arr = np.genfromtxt(f,
                    delimiter=",", dtype=int)
            lst = arr.tolist() 
            lst = [0 if item == -1 else item for row in lst for item in row]
            arr2 = np.array(lst,dtype=object)
            arr2 = np.reshape(arr2,[-1,9])
            lst_board.append(arr2)
    board_data = np.vstack(lst_board)
    df = pd.DataFrame(columns=["board","astar","b_count"])
    df['board'] = board_data.tolist()
    
    lst_astar_filename = ['3x3_1BLANK_10000_COSTS-1.csv','3x3_2BLANK_10000_COSTS.csv']
    lst_astar = []
    for filename in lst_astar_filename:
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and 'csv' in filename:
            arr = np.genfromtxt(f,
                    delimiter=",", dtype=int)
            lst_astar.append(arr)
    astar_data = np.concatenate(lst_astar)
    df['astar'] = astar_data.tolist()
    
    df['b_count'] = df['board'].apply(count_B)
    return df

if __name__ == "__main__":
    print(parse_board())
    # print(read_aggregated())