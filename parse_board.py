import csv
import os
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

    return board_df
    
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
                data.append(element)
                B_count += 1
    board_data = data[:-1]
    a_star = data[-1]
    return board_data,a_star,B_count

if __name__ == "__main__":
    parse_board()