from random import randint


def generate_permutation(n):
    repres = [1]
    while len(repres) < n:
        x = randint(2, n)
        if x not in repres:
            repres.append(x)
    repres.append(1)
    return repres


def fitness(chromosome, matrix):
    fit = 0
    for i in range(len(chromosome) - 1):
        fit += matrix[chromosome[i] - 1][chromosome[i + 1] - 1]
    return fit


class Chromosome:
    def __init__(self, problParam=None):
        self.__problParam = problParam
        self.__repres = generate_permutation(problParam['noNodes'])
        self.__fitness = 0.0

    @property
    def repres(self):
        return self.__repres

    @property
    def fitness(self):
        return self.__fitness

    @repres.setter
    def repres(self, l=None):
        if l is None:
            l = []
        self.__repres = l

    @fitness.setter
    def fitness(self, fit=0.0):
        self.__fitness = fit

    def crossover(self, c):
        pos1 = randint(1, self.__problParam['noNodes'] - 1)
        pos2 = randint(1, self.__problParam['noNodes'] - 1)
        if pos2 < pos1:
            pos1, pos2 = pos2, pos1
        k = 0
        newrepres = self.__repres[pos1: pos2 + 1]
        for el in c.__repres[pos2 + 1:-1] + c.__repres[1:pos2 + 1]:
            if el not in newrepres:
                if len(newrepres) < self.__problParam['noNodes'] - pos1:
                    newrepres.append(el)
                else:
                    newrepres.insert(k, el)
                    k += 1
        path = [1] + newrepres + [1]
        offspring = Chromosome(self.__problParam)
        offspring.__repres = path
        return offspring

    def mutation(self):
        pos1 = randint(1, self.__problParam['noNodes'] - 1)
        pos2 = randint(1, self.__problParam['noNodes'] - 1)
        self.__repres[pos1], self.__repres[pos2] = self.__repres[pos2], self.__repres[pos1]

    def __str__(self):
        return "\nCromozom: " + str(self.__repres) + " fitness: " + str(self.__fitness)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, c):
        return self.__repres == c.__repres and self.__fitness == c.__fitness
