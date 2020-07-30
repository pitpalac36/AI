import numpy as np
from Colony import Colony, print_route
import fileUtils


def run():
    distances = np.array(fileUtils.read_from_directory(5))
    ant_colony = Colony(distances, 10, 100)
    ant_colony.apply_perturbation()
    shortest_path = ant_colony.run()
    print_route(shortest_path, "\nRuta cea mai buna : ")


run()
