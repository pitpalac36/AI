import os
import numpy


def distance(x1, x2, y1, y2):
    return numpy.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def read_from_file(file_path):
    f = open(file_path, "r")
    n = int(f.readline())
    matrix = []
    for i in range(n):
        matrix.append([])
        line = f.readline()
        elements = line.split(",")
        for k in range(n):
            if int(elements[k]) == 0:
                matrix[-1].append(numpy.inf)
            else:
                matrix[-1].append(int(elements[k]))
    f.close()
    return matrix


def read_coordinates_from_file(file_path):
    f = open(file_path, "r")
    lines = f.readlines()
    matrix = []
    coords = []
    for i in range(7, len(lines) - 1):
        one = lines[i].split(" ")
        coords.append(one)
    for i in range(len(coords)):
        matrix.append([])
        for j in range(len(coords)):
            dist = distance(float(coords[i][1]), float(coords[j][1]), float(coords[i][2]), float(coords[j][2]))
            if dist == 0:
                matrix[i].append(numpy.inf)
            else:
                matrix[i].append(dist)
    f.close()
    return matrix


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
