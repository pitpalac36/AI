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
    age = [int(data[i][dataNames.index('age')]) for i in range(len(data))]
    height = [int(data[i][dataNames.index('height')]) for i in range(len(data))]
    weight = [int(data[i][dataNames.index('weight')]) for i in range(len(data))]
    sex = [data[i][dataNames.index('sex')] for i in range(len(data))]
    return age, height, weight, sex

