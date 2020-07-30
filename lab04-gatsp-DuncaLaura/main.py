import fileUtils
from GA import GA
from Chromosome import fitness


def main():
    network = fileUtils.read_from_directory(4)
    gaParam = {'size': 100, 'generations': 100}
    problParam = {'noNodes': network['noNodes'], 'matrix': network['matrix'], 'function': fitness}
    ga = GA(gaParam, problParam)
    ga.initialization()
    ga.evaluation()
    for generation in range(gaParam['generations']):
        # ga.one_generation_steady_state()
        # ga.one_generation_elitism()
        ga.one_generation_elitism_improved()
        best = ga.best_chromosome()
        print('Solutia optima in generatia ' + str(generation + 1) + ' este ' + str(best.repres) +
              'fitness = ' + str(best.fitness))
    best = ga.best_chromosome()
    print('\nSolutia optima : ' + str(best.repres) + '\n                 fitness = ' + str(best.fitness))


main()
