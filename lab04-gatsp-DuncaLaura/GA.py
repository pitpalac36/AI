from random import randint
from Chromosome import Chromosome


def best_chromosome_from_another_population(population):
    best = population[0]
    for each in population:
        if each.fitness < best.fitness:
            best = each
    return best


class GA:
    def __init__(self, param=None, problParam=None):
        self.__param = param
        self.__problParam = problParam
        self.__population = []

    @property
    def population(self):
        return self.__population

    def initialization(self):
        for i in range(0, self.__param['size']):
            c = Chromosome(self.__problParam)
            self.__population.append(c)

    def evaluation(self):
        for c in self.__population:
            c.fitness = self.__problParam['function'](c.repres, self.__problParam['matrix'])

    def best_chromosome(self):
        best = self.__population[0]
        for each in self.__population:
            if each.fitness < best.fitness:
                best = each
        return best

    def worst_chromosome(self):
        worst = self.__population[0]
        for each in self.__population:
            if each.fitness > worst.fitness:
                worst = each
        return worst

    def selection(self):
        poz1 = randint(0, self.__param['size'] - 1)
        poz2 = randint(0, self.__param['size'] - 1)
        if self.__population[poz1].fitness < self.__population[poz2].fitness:
            return poz1
        else:
            return poz2

    def one_generation_steady_state(self):
        for _ in range(self.__param['size']):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            offspring = p1.crossover(p2)
            offspring.mutation()
            offspring.fitness = self.__problParam['function'](offspring.repres, self.__problParam['matrix'])
            worst = self.worst_chromosome()
            if offspring.fitness < worst.fitness:
                for i in range(self.__param['size']):
                    if self.__population[i] == worst:
                        self.__population[i] = offspring
                        break

    def one_generation_elitism(self):
        bestSamples = [self.best_chromosome()]
        for _ in range(self.__param['size'] - 1):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            offspring = p1.crossover(p2)
            offspring.mutation()
            offspring.fitness = self.__problParam['function'](offspring.repres, self.__problParam['matrix'])
            if offspring.fitness < p1.fitness and offspring.fitness < p2.fitness:
                bestSamples.append(offspring)
            else:
                bestSamples.append(p1)
                bestSamples.append(p2)
        self.__population = bestSamples
        self.evaluation()

    def one_generation_elitism_improved(self):
        bestSamples = [self.best_chromosome()]
        for _ in range(self.__param['size'] - 1):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            offsprings = []
            for i in range(0, 3):
                one = p1.crossover(p2)
                one.mutation()
                one.fitness = self.__problParam['function'](one.repres, self.__problParam['matrix'])
                offsprings.append(one)
            bestOne = best_chromosome_from_another_population(offsprings)
            if bestOne.fitness < p1.fitness and bestOne.fitness < p2.fitness:
                bestSamples.append(bestOne)
            else:
                bestSamples.append(p1)
                bestSamples.append(p2)
        self.__population = bestSamples
        self.evaluation()
