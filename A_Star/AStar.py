########
## Group 16 Astar code with alterations from group 17
########

from Algorithm import Algorithm
from queue import PriorityQueue
import time


class AStar(Algorithm):

    def __init__(self, board, heuristic, weighted):
        super().__init__(board, heuristic, weighted)
        self.nodes_expanded = 0
        self.elapsed_time = None
        self.greedy_cache = dict()

    def start(self):
        print(f'Performing A* search with {self.heuristic_type} heuristic {"with" if self.weighted else "without"} weight')
        print("Initial Board:")

        print(self.board)
        end = self.search()

        self._print_path(end)

    def search(self):
        total_runs = 0
        total_time = 0
        fringe = PriorityQueue()
        heuristic, calc_time = self._calculate_heuristic(self.board)
        total_runs+=1
        total_time+=calc_time
        fringe.put((heuristic, self.board))
        visited = dict()

        start_time = time.time()
        while not fringe.empty():
            current = fringe.get()[1]

            self.nodes_expanded += 1

            heuristic, calc_time = self._calculate_heuristic(current)
            total_runs+=1
            total_time+=calc_time
            
            if heuristic == 0:
                print("Average heuristic calculation time:")
                print(total_time/total_runs)
                self.elapsed_time = time.time() - start_time
                return current

            for board in current.neighbors():
                if board not in visited or board.cost < visited[board]:
                    visited[current] = current.cost
                    heuristic, calc_time = self._calculate_heuristic(board)
                    total_time+=calc_time
                    total_runs+=1
                    priority = board.cost + heuristic
                    fringe.put((priority, board))

    def _create_path_string(self, goal):
        current = goal
        moves = []

        while current.previous is not None:
            moves.append("{} {}".format(current.value, current.direction.name))
            current = current.previous

        moves.reverse()
        return moves

    def _print_path(self, end):
        path = self._create_path_string(end)

        print("\nPath:")
        print("\n".join(path))

        print("\nFinal Board State:")
        print(end)

        print("\nCost: {}".format(end.cost))
        print("Length: {}".format(len(path)))

        print("Nodes Expanded: {}".format(self.nodes_expanded))
        print("Estimated Branching Factor: {}".format(pow(self.nodes_expanded, 1 / len(path))))
        print("Elapsed Time: {}".format(self.elapsed_time))

        print()
