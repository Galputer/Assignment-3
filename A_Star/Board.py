########
## Group 16 Astar code with alterations from group 17
########

import numpy as np
from Algorithm import Direction
from copy import deepcopy


class Board:
    def __init__(self, board, previous=None, cost=0, direction=None, value=None):
        self.board = board
        self.previous = previous
        self.cost = cost
        self.direction = direction
        self.value = value

    def neighbors(self):
        neighbors = []
        for row in range(len(self.board)):
            for col in range(len(self.board)):
                if self.board[row][col] == 0:
                    neighbors = neighbors + self._neighbor(row, col)
        return neighbors

    def _neighbor(self, row, col):
        neighbors = []
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for r, c in directions:
            if self._is_legal(row + r, col + c) and self.board[row + r][col + c] != 0:
                new_board = self._swap(deepcopy(self.board), row + r, col + c, row, col)
                board = Board(new_board, self, self.cost + new_board[row][col], Direction((-r, -c)), new_board[row][col])
                neighbors.append(board)
        return neighbors

    def _swap(self, board, row1, col1, row2, col2):
        board[row1][col1], board[row2][col2] = board[row2][col2], board[row1][col1]
        return board

    def _is_legal(self, row, col):
        return 0 <= row < len(self.board) and 0 <= col < len(self.board)

    def __hash__(self):
        string = ""
        for row in self.board:
            for num in row:
                string += str(num)
        return hash(string)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return (self.board == other.board).all()

    def __lt__(self, other):
        np.less(self.board, other.board)

    # allows Board to be treated as a normal 2d array
    def __getitem__(self, item):
        return self.board[item]

    def __iter__(self):
        return self.board.__iter__()

    def __len__(self):
        return len(self.board)

    def __str__(self):
        return '\n'.join(map(str, self.board))
