import csv


def read_from_csv(fileName):
    data = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    temp = [int(data[i][dataNames.index('temperature')]) for i in range(len(data))]
    female = [int(data[i][dataNames.index('femaleTshirts')]) for i in range(len(data))]
    male = [int(data[i][dataNames.index('maleTshirts')]) for i in range(len(data))]
    return temp, female, male


def read_from_csv2(fileName):
    data = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    temp = [int(data[i][dataNames.index('temperature')]) for i in range(len(data))]
    female = [int(data[i][dataNames.index('femaleTshirts')]) for i in range(len(data))]
    male = [int(data[i][dataNames.index('maleTshirts')]) for i in range(len(data))]
    competitions = [data[i][dataNames.index('competitions')] for i in range(len(data))]
    location = [data[i][dataNames.index('location')] for i in range(len(data))]
    return temp, female, male, competitions, location
