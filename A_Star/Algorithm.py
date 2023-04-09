########
## Group 16 Astar code with alterations from group 17
########

from abc import abstractmethod, ABC
from enum import Enum
import os
from pathlib import Path
import pickle
import time

HEURISTIC_TELEPORT = "teleporting"
HEURISTIC_SLIDE = "sliding"
HEURISTIC_LEARNED = "learned"


class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    def __str__(self):
        direction = ''

        if self == Direction.UP:
            direction = "UP"
        elif self == Direction.DOWN:
            direction = "DOWN"
        elif self == Direction.LEFT:
            direction = "LEFT"
        elif self == Direction.RIGHT:
            direction = "RIGHT"

        return direction


class Move:
    def __init__(self, value, direction):
        self.direction = direction
        self.value = value

    def __str__(self):
        return f'{self.value} {self.direction}'


class Algorithm(ABC):

    def __init__(self, board, heuristic, weighted):
        self.board = board
        self.heuristic_type = heuristic
        self.weighted = weighted
        self.goal_state_front_blanks, self.goal_state_back_blanks = self.goal_state(self.board)
        self.learned_model = self.load_model()

    # Driver method for algorithms
    @abstractmethod
    def start(self):
        pass
    
    def load_model(self):
        model_path = Path(os.path.dirname(os.path.realpath(__file__))).parent / "models" / "lasso-3.pkl"
        loaded_model = pickle.load(open(model_path, 'rb'))
        return loaded_model
    
    # to be run once on init, and never again
    def goal_state(self, board):
        arr = []
        num_of_0 = 0
        for x in range(len(board)):
            for y in range(len(board[x])):
                if board[x][y] == 0:
                    num_of_0 += 1
                else:
                    arr.append(board[x][y])
        arr.sort()

        back_dict = {}
        front_dict = {}

        for x in range(len(arr)):
            back_dict[arr[x]] = (x // len(board), x % len(board))
            front_dict[arr[x]] = ((x + num_of_0) // len(board), (x + num_of_0) % len(board))

        return front_dict, back_dict

    # Heuristic functions:
    def _calculate_heuristic(self, board):
        if self.heuristic_type == HEURISTIC_TELEPORT:
            front_heuristic, back_heuristic, calc_time = self._calculate_teleport_heuristic(board)
            return min(front_heuristic, back_heuristic), calc_time
        elif self.heuristic_type == HEURISTIC_SLIDE:
            front_heuristic, back_heuristic, calc_time = self._calculate_slide_heuristic(board)
            return min(front_heuristic, back_heuristic), calc_time
        elif self.heuristic_type == HEURISTIC_LEARNED:
            front_heuristic, back_heuristic, calc_time = self._calculate_learned_heuristic(board, self.learned_model)
            return min(front_heuristic, back_heuristic), calc_time
        else:
            front_heuristic, back_heuristic, calc_time = self._calculate_learned_heuristic(board, self.learned_model)
            return min(front_heuristic, back_heuristic), calc_time

    def _calculate_learned_heuristic(self, board, model):
        start_time = time.time()
        board1d = [j for sub in board for j in sub]
        # ['incorrect_count','inversion','incorrect_sum','manhattan','conflicts','hamming']
        X_front = []
        X_back = []
        temp1 = self._count_incorrect_tiles(board, self.goal_state_back_blanks)
        temp2 = self._count_incorrect_tiles(board, self.goal_state_front_blanks)
        if temp1 == 0 or temp2 == 0: return temp1, temp2, time.time()-start_time
        inv = self._inversions_count(board1d)
        X_back.append(inv)
        X_front.append(inv)
        X_back.append(self._sum_incorrect_weight(board, self.goal_state_back_blanks))
        X_front.append(self._sum_incorrect_weight(board, self.goal_state_front_blanks))
        manhattan_front, manhattan_back, _ = self._calculate_slide_heuristic(board)
        X_back.append(manhattan_back)
        X_front.append(manhattan_front)
        X_front.append(manhattan_front + self._linear_conflict(board, self.goal_state_front_blanks))
        X_back.append(manhattan_back + self._linear_conflict(board, self.goal_state_back_blanks))

        X_back.append(self._hamming_distance(board, self.goal_state_back_blanks))
        X_front.append(self._hamming_distance(board, self.goal_state_front_blanks))

        y = model.predict([X_front,X_back])
        y_front = y[0]
        y_back = y[1]
        return y_front, y_back, time.time()-start_time
    
    def _calculate_teleport_heuristic(self, board):
        start_time = time.time()
        front_heuristic = 0
        back_heuristic = 0
        for x in range(len(board)):
            for y in range(len(board[x])):
                current = board[x][y]
                if current == 0:
                    continue
                front_blank_coordinates = self.goal_state_front_blanks[current]
                back_blank_coordinates = self.goal_state_back_blanks[current]
                if (x, y) != front_blank_coordinates:
                    front_heuristic += 1 * (current if self.weighted else 1)
                if (x, y) != back_blank_coordinates:
                    back_heuristic += 1 * (current if self.weighted else 1)
        return front_heuristic, back_heuristic, time.time()-start_time

    def _calculate_slide_heuristic(self, board):
        start_time = time.time()
        front_heuristic = 0
        back_heuristic = 0
        for x in range(len(board)):
            for y in range(len(board[x])):
                current = board[x][y]
                if current == 0:
                    continue
                front_heuristic += self._manhattan_distance_to_goal((x, y), current, True) * (
                    current if self.weighted else 1)
                back_heuristic += self._manhattan_distance_to_goal((x, y), current, False) * (
                    current if self.weighted else 1)
        return front_heuristic, back_heuristic, time.time()-start_time

    def _manhattan_distance_to_goal(self, location, value, front):
        location_2 = self.goal_state_front_blanks[value] if front else self.goal_state_back_blanks[value]
        return abs(location[0] - location_2[0]) + abs(location[1] - location_2[1])
    
    def _count_incorrect_tiles(self, board, goal):
        count = 0

        for i in range(len(board)):
            for j in range(len(board[i])):
                cur_tile = board[i][j]
                if cur_tile == 0: continue
                if i != goal[cur_tile][0] or j != goal[cur_tile][1]: count += 1
            
        return count
    
    def _inversions_count(self, board):
        """
        Calculates the number of inversions in the given board state.
        """
        inversions = 0
        for i in range(len(board)):
            for j in range(i + 1, len(board)):
                if board[i] != 0 and board[j] != 0 and board[i] > board[j]:
                    inversions += 1
        return inversions
    
    def _sum_incorrect_weight(self, board, goal):
        sum = 0

        for i in range(len(board)):
            for j in range(len(board[i])):
                cur_tile = board[i][j]
                if cur_tile == 0 : continue
                if (i,j) != goal[cur_tile]: sum += cur_tile
            
        return sum
    
    def _hamming_distance(self, board, goal):
        """
        Calculates the Hamming distance between two boards.
        """
        distance = 0
        for i in range(len(board)):
            for j in range(len(board[i])):
                cur_tile = board[i][j]
                if cur_tile == 0: continue
                if (i,j) != goal[cur_tile]:
                    distance += cur_tile
        return distance
    
    def _linear_conflict(self, board, goal):
        size = len(board)
        conflict = 0

        # Calculate conflict for rows
        for i in range(size):
            for j in range(size):
                if board[i][j] != 0 and (i,j) != goal[board[i][j]]:
                    if i == goal[board[i][j]][0]:
                        for k in range(j + 1, size):
                            if board[i][k] != 0 and (i,k) != goal[board[i][j]] and goal[board[i][j]][1] < goal[board[i][k]][1]:
                                conflict += board[i][j] + board[i][k]

        # Calculate conflict for columns
        for j in range(size):
            for i in range(size):
                if board[i][j] != 0 and (i,j) != goal[board[i][j]]:
                    if j == goal[board[i][j]][1]:
                        for k in range(i + 1, size):
                            if board[k][j] != 0 and (k,j) != goal[board[i][j]] and goal[board[i][j]][0] < goal[board[k][j]][0]:
                                conflict += board[i][j] + board[k][j]

        return conflict
