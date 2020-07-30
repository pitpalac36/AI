import random
from math import sqrt
import numpy as np


def euclidean_distance(x1, x2):
    squared_distance = 0
    for i in range(len(x1)):
        squared_distance += (x1[i] - x2[i]) ** 2
    return sqrt(squared_distance)


def jaccard_similarity(x1, x2):
    intersect = 0
    for i in range(len(x1)):
        if x1[i] == x2[i] and x1[i] != 0:
            intersect += x1[i]
    reunion = len(x1) + len(x2) - intersect
    return intersect / reunion


class kmeans:

    def __init__(self, k, similarity, max_iterations):
        self.k = k
        self.iterations = max_iterations
        self.centroids = [[] for _ in range(self.k)]
        self.clusters = {}
        self.similarity = similarity

    def fit(self, data):

        for i in range(self.k):
            self.centroids[i] = random.choice(data)

        for i in range(self.iterations):
            self.clusters = {}
            for j in range(self.k):
                self.clusters[j] = []

            for each in data:
                distances = [self.similarity(centroid, each) for centroid in self.centroids]
                # print(distances)
                if self.similarity == euclidean_distance:
                    dist = distances.index(min(distances))
                if self.similarity == jaccard_similarity:
                    dist = distances.index(max(distances))
                self.clusters[dist].append(each)

            # ACTUALIZAREA CENTROIZILOR
            if self.similarity == euclidean_distance or self.similarity == jaccard_similarity:   # actualizez centroizii cu mediile clusterilor
                for each in self.clusters:
                    self.centroids[each] = np.average(self.clusters[each], axis=0)

            if self.similarity == jaccard_similarity:   # actualizez cu valorile cele mai apropiate de medii
                mean = {}
                for l in range(len(self.clusters)):
                    dis = {}
                    mean[l] = np.average(self.clusters[l], axis=0)
                    for one in self.clusters[l]:
                        dis[jaccard_similarity(one, mean[l])] = one
                    maximum = max(dis)
                    self.centroids[l] = dis.get(maximum)   # valoarea din dictionar pt care cheia (similaritatea) este maxima


    def predictEuclidean(self, sample):
        centroid = random.choice(self.centroids)
        shortest_distance = 100000
        for each in self.centroids:
            if self.similarity(each, sample) < shortest_distance:
                shortest_distance = self.similarity(each, sample)
                centroid = each
        return centroid


    def predictJaccard(self, sample):
        centroid = self.centroids[0]
        most_similar = -1
        for each in self.centroids:
            if self.similarity(each, sample) > most_similar:
                most_similar = self.similarity(each, sample)
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
