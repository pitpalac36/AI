import random
from math import sqrt
import numpy as np


# durata medie de comunicare
def euclidean_distance(x1, x2):
    squared_distance = (x1[3] - x2[3]) ** 2
    return sqrt(squared_distance)


class kmeans:

    def __init__(self, k, similarity = euclidean_distance, max_iterations=100):
        self.k = k
        self.iterations = max_iterations
        self.centroids = [[] for _ in range(self.k)]
        self.clusters = {}
        self.similarity = similarity

    def fit(self, data):

        self.centroids[1] = [data[i] for i in range(len(data)) if data[i][3] == 15][0]
        self.centroids[0] = [data[i] for i in range(len(data)) if data[i][3] == 92][0]
        for i in range(self.iterations):
            self.clusters = {}
            for j in range(self.k):
                self.clusters[j] = []

            for each in data:
                distances = [self.similarity(centroid, each) for centroid in self.centroids]
                # print(distances)
                if self.similarity == euclidean_distance:
                    dist = distances.index(min(distances))
                self.clusters[dist].append(each)

            # ACTUALIZAREA CENTROIZILOR
            if self.similarity == euclidean_distance:   # actualizez centroizii cu mediile clusterilor
                for each in self.clusters:
                    mean = 0
                    for one in self.clusters[each]:
                        mean += one[3]
                    mean = mean / len(self.clusters[each])
                    self.centroids[each][3] = mean


    def fit_trained_already(self, dataNew):
        for each in dataNew:
            distances = [self.similarity(centroid, each) for centroid in self.centroids]
            # print(distances)
            if self.similarity == euclidean_distance:
                dist = distances.index(min(distances))
            self.clusters[dist].append(each)


    def predictEuclidean(self, sample):
        centroid = random.choice(self.centroids)
        shortest_distance = 100000
        for each in self.centroids:
            if self.similarity(each, sample) < shortest_distance:
                shortest_distance = self.similarity(each, sample)
                centroid = each
        return centroid


    def dunn_index(self):
        intercluster_distances = []
        intracluster_distances = []
        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                intercluster_distances.append(self.similarity(self.centroids[i], self.centroids[j]))

            for a in self.clusters[i]:
                for b in self.clusters[i]:
                    intracluster_distances.append(self.similarity(a, b))

        return min(intercluster_distances) / max(intracluster_distances)
