########
## Group 16 Astar code with alterations from group 17
########

from abc import abstractmethod, ABC
from enum import Enum

HEURISTIC_TELEPORT = "teleporting"
HEURISTIC_SLIDE = "sliding"


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

    # Driver method for algorithms
    @abstractmethod
    def start(self):
        pass

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
            return min(self._calculate_teleport_heuristic(board))
        elif self.heuristic_type == HEURISTIC_SLIDE:
            return min(self._calculate_slide_heuristic(board))
        else:
            return min(self._calculate_slide_heuristic(board))

    def _calculate_teleport_heuristic(self, board):
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
        return front_heuristic, back_heuristic

    def _calculate_slide_heuristic(self, board):
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
        return front_heuristic, back_heuristic

    def _manhattan_distance_to_goal(self, location, value, front):
        location_2 = self.goal_state_front_blanks[value] if front else self.goal_state_back_blanks[value]
        return abs(location[0] - location_2[0]) + abs(location[1] - location_2[1])

        #Cooper's addition, serpent feature
    
    def adjasentToPreAndSubsequentTiles(self, location, value, front):
        #relPos = [-1,0,1]
        #relRow = -1
        preAndSubsequentAdj = 0
        neighbors = self._neighbors(location[0],location[1])
        for adjasent in neighbors:
            if value == neighbors[adjasent].value + 1:
                preAndSubsequentAdj = preAndSubsequentAdj + 1
            if value == neighbors[adjasent].value - 1:
                preAndSubsequentAdj = preAndSubsequentAdj + 1
        return preAndSubsequentAdj

