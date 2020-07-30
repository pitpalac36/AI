import random

import numpy as np


# from a list of pairs to an actual path
def print_route(path, message):
    pairs = path[0]
    cost = path[1]
    print(message, pairs[0][0], "", pairs[0][1], end="")
    for i in range(1, len(pairs)):
        print(" ", pairs[i][1], end="")
    print("     cost : ", cost)


class Colony:

    def __init__(self, matrix, noAnts, iterations):
        self.__matrix = matrix
        self.__noAnts = noAnts
        self.__iterations = iterations
        self.__phi = 0.95
        self.__alpha = 1
        self.__beta = 1
        self.__pheromone = np.ones(self.__matrix.shape)

    def apply_perturbation(self):
        number_of_edges = random.randint(1, len(self.__matrix) // 2)
        print("Se aplica perturbare pe {} muchii".format(number_of_edges))
        for i in range(number_of_edges):
            first_node = int(random.uniform(0, len(self.__matrix)))
            second_node = int(random.uniform(0, len(self.__matrix)))
            if first_node != second_node:
                old_cost = self.__matrix[first_node][second_node]
                self.__matrix[first_node][second_node] = self.__matrix[first_node][second_node] * random.uniform(0.5, 1.5)
                self.__matrix[second_node][first_node] = self.__matrix[first_node][second_node]
                print("Am modificat muchia {} -> {}, cost initial: {}, cost perturbat :{}\n"
                      .format(first_node, second_node, old_cost, self.__matrix[first_node][second_node]))
            else:
                number_of_edges += 1

    def spread_pheromone(self, all_paths, no_ants_which_spread):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:no_ants_which_spread]:
            for move in path:
                self.__pheromone[move] += 1.0 / self.__matrix[move]

    def choose_next(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        decision_quality = pheromone ** self.__alpha * ((1.0 / dist) ** self.__beta)
        probability = decision_quality / decision_quality.sum()
        move = np.random.choice(len(self.__matrix), 1, p=probability)[0]
        return move

    def get_path_cost(self, path):
        return sum(self.__matrix[elem] for elem in path)

    def generate_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.__matrix) - 1):
            move = self.choose_next(self.__pheromone[prev], self.__matrix[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))
        return path

    def generate_paths(self):
        all_paths = []
        for i in range(self.__noAnts):
            path = self.generate_path(1)
            all_paths.append((path, self.get_path_cost(path)))
        return all_paths

    def run(self):
        all_time_shortest_path = ("", np.inf)
        for i in range(self.__iterations):
            all_paths = self.generate_paths()
            self.spread_pheromone(all_paths, self.__noAnts)
            shortest_path = min(all_paths, key=lambda x: x[1])
            print_route(shortest_path, "Iteratia " + str(i + 1) + " : ")
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.__pheromone *= self.__phi
        return all_time_shortest_path
