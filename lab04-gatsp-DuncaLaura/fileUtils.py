import os
import numpy


def read_from_file(filepath):
    f = open(filepath, "r")
    network = {}
    n = int(f.readline())
    network['noNodes'] = n
    matrix = []
    for i in range(n):
        matrix.append([])
        line = f.readline()
        elements = line.split(",")
        for k in range(n):
            matrix[-1].append(int(elements[k]))
    network['matrix'] = matrix
    f.close()
    return network


def euclidean_distance(x1, x2, y1, y2):
    return numpy.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def read_coordinates_from_file(file_path):
    f = open(file_path, "r")
    lines = f.readlines()
    network = {}
    matrix = []
    coords = []
    for i in range(7, len(lines) - 1):
        one = lines[i].split(" ")
        coords.append(one)
    for i in range(len(coords)):
        matrix.append([])
        for j in range(len(coords)):
            distance = euclidean_distance(float(coords[i][1]), float(coords[j][1]),
                                          float(coords[i][2]), float(coords[j][2]))
            if distance == 0:
                matrix[i].append(numpy.inf)
            else:
                matrix[i].append(distance)
    f.close()
    network['noNodes'] = len(matrix)
    network['matrix'] = matrix
    return network


def read_from_directory(index):
    files = []
    directory = "data/"
    for entry in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, entry)):
            files.append(entry)
    if files[index].__contains__("easy") or files[index].__contains__("medium"):
        network = read_from_file(directory + files[index])
    else:
        network = read_coordinates_from_file(directory + files[index])
    return network
