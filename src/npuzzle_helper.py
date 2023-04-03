import copy
import csv
import itertools

def read_file(file_path):
    """Read the board file from csv to memory.
    Return the board content and number of B letters.

    Args:
        file_path (String): The path string for input board csv file

    Returns:
        data: 2D array of board file
        B_count: number of letter B in the board
    """
    datafile = open(file_path, 'r', encoding="utf-8-sig")
    datareader = csv.reader(datafile, delimiter=',')
    data = []
    B_count = 0
    for row in datareader:
        cur_row = []
        for element in row:
            if element.isnumeric():
                cur_row.append(int(element))
            else:
                cur_row.append(element)
                B_count += 1
        data.append(cur_row)    

    return data, B_count

def goal_states(board,B_count):
    """Generate the goal states for the current board.
    Also generate dictionaries of those goal states.

    Args:
        board (2D array): content of the board
        B_count (int): number of letter B in the board

    Returns:
        goal_board1_2d: goal board with B in the front
        goal_board2_2d: goal board with B at the end
        dict1: dictionary representation of goal_board1
        dict2: dictionary representation of goal_board2
    """
    n=len(board)
    board_1d = list(itertools.chain.from_iterable(board))
    board_1d = sorted([e for e in board_1d if e != 'B'])
    list_of_B = ['B'] * B_count
    goal_board1_1d = list_of_B + board_1d
    goal_board2_1d = board_1d + list_of_B
    goal_board1_2d = []
    goal_board2_2d = []
    for i in range(n):
        goal_board1_2d.append(goal_board1_1d[i*n:i*n+n])
        goal_board2_2d.append(goal_board2_1d[i*n:i*n+n])
    # print('goal_board1_2d',goal_board1_2d) 
    # print('goal_board2_2d',goal_board2_2d) 
    
    dict1 = goal_board_dicts(goal_board1_2d)
    dict2 = goal_board_dicts(goal_board2_2d)
    return goal_board1_2d, goal_board2_2d, dict1, dict2

def goal_board_dicts(goal_board):
    """Helper function to generate dictionary representation of goal board

    Args:
        goal_board (2D array): content of the goal board
    """
    n=len(goal_board)
    dict_goal = {}
    for row in range(n):
        for col in range(n):
            cur_tile = goal_board[row][col]
            if isinstance(cur_tile, str):
                continue
            dict_goal[cur_tile] = (row,col)
    # print('dict_goal',dict_goal)
    return(dict_goal)


def valid_moves(board):
    """Get the valid moves for the current board.

    Args:
        board (2D array): content of the board

    Returns:
        list: list of valid moves on the board.
        Each entry in the list is a list of two parts.
        The first part is the number e.g. 1-9.
        The second part is the move direction in form of (dx,dy)
    """
    n = len(board)
    list_of_B_coords = []
    for row in range(n):
        for col in range(n):
            cur_tile = board[row][col]
            if not isinstance(cur_tile, str):
                continue
            list_of_B_coords.append((row,col))
    # print('list_of_B_coords',list_of_B_coords)
    list_of_moves = []
    for (row_cur,col_cur) in list_of_B_coords:
        for (dx, dy) in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            if inbound(row_cur + dx, col_cur + dy, n):
                if not isinstance(board[row_cur + dx][col_cur + dy],str):
                    list_of_moves.append([board[row_cur + dx][col_cur + dy], (0 - dx, 0 - dy)])
    # print('list_of_moves',list_of_moves)
    return list_of_moves

def inbound(row,col,n):
    """check if the coordinate is within the nxn matrix

    Args:
        row (int): row index
        col (int): col index
        n (int): n of the nxn board

    Returns:
        boolean : True if the coordinate is inbound,
        False otherwise
    """
    return 0<=row<n and 0<=col<n

def get_moved_board(board, move):
    """get a new board from the previous board
    and the given move of the tile

    Args:
        board (2D array): Current board layout
        move (list): Move is a list of two parts.
        The first part is the number e.g. 1-9.
        The second part is the move direction in form of (dx,dy)
    """
    tile = move[0]
    (dx,dy) = move[1]
    n = len(board)
    tile_x = -1
    tile_y = -1
    for row in range(n):
        break_flag = False
        for col in range(n):
            cur_tile = board[row][col]
            if not (tile == cur_tile):
                continue
            tile_x = row
            tile_y = col
            break_flag = True
            break
        if break_flag: break
    
    modified_board = copy.deepcopy(board)

    modified_board[tile_x][tile_y] = 'B'
    modified_board[tile_x+dx][tile_y+dy] = tile
    # print("modified_board",modified_board)
    return modified_board

def move_to_string(move):
    """change the list formate move to readable string

    Args:
        move (list): Move is a list of two parts.
        The first part is the number e.g. 1-9.
        The second part is the move direction in form of (dx,dy)

    Returns:
        string: string representation of move
    """
    (tile,direction) = move
    move_rep = {
        (0,1): 'right',
        (0,-1): 'left',
        (1,0): 'down',
        (-1,0): 'up'
    }
    direction_str = move_rep[direction]
    return str(tile) + ' ' + direction_str

def print_board(board):
    n = len(board)
    for x in range(n):
        print("[",end=" ")
        for y in range(n):
            if isinstance(board[x][y],str):
                print(' B',end=" ")
            else:
                if board[x][y] > 9:
                    print(str(board[x][y]),end=" ")
                else:
                    print(" " + str(board[x][y]),end=" ")
        print("]")