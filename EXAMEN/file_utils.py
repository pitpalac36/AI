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
    id = [data[i][dataNames.index('clientID')] for i in range(len(data))]
    ultima_com = [data[i][dataNames.index('lastCommunicTime')] for i in range(len(data))]
    information = [data[i][dataNames.index('information')] for i in range(len(data))]
    averageCommunicLength = [int(data[i][dataNames.index('averageCommunicLength')]) for i in range(len(data))]
    clientSpecialisation = [data[i][dataNames.index('clientSpecialisation')] for i in range(len(data))]
    return id, ultima_com, information, averageCommunicLength, clientSpecialisation
