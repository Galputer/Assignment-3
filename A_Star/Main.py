########
## Group 16 Astar code with alterations from group 17
########

import pandas as pd
import sys

import Algorithm
import AStar
from Board import Board

COMMAND_ARGUMENT_ASTAR = "npuzzle"


def main(argv):

    command_format = "\ncommands: [npuzzle] [board file] [sliding/teleporting/greedy] [true/false]"

    print(command_format)

    while True:
        argv = get_input()

        if len(argv) != 0 and argv[0] == "q":
            return

        if len(argv) < 3:
            print("Not enough command line arguments", command_format)
            continue

        board = None

        try:
            directory = './Boards/' + argv[1]
            board = Board(pd.read_csv(directory, sep=',', header=None).replace('B', 0).values.astype(int))

        except Exception as e:
            print(e, command_format)
            continue

        algorithm = argv[0]

        driver = None

        if algorithm == COMMAND_ARGUMENT_ASTAR:

            heuristic = argv[2]
            bool = argv[3].lower()

            weight = None

            if bool == "true":
                weight = True
            elif bool == "false":
                weight = False

            if not heuristic.lower() in [Algorithm.HEURISTIC_TELEPORT, Algorithm.HEURISTIC_SLIDE]:
                print("Unknown heuristic for command argument 2", command_format)
                continue

            driver = AStar.AStar(board, heuristic, weight)
        else:
            print("Unknown algorithm for command argument 1", command_format)
            continue

        driver.start()


def get_input():
    user_input = input("Enter arguments:")
    return user_input.split()


if __name__ == "__main__":
    main(sys.argv[1:])
