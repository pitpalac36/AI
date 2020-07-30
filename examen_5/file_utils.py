import csv


def read_from_csv1(fileName):
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
    density = [float(data[i][dataNames.index('density')]) for i in range(len(data))]
    alcool = [float(data[i][dataNames.index('alcohol')]) for i in range(len(data))]
    return density, alcool


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
    fixed_acidity = [float(data[i][dataNames.index('fixed acidity')]) for i in range(len(data))]
    volatile_acidity = [float(data[i][dataNames.index('volatile acidity')]) for i in range(len(data))]
    citric_acid = [float(data[i][dataNames.index('citric acid')]) for i in range(len(data))]
    residual_sugar = [float(data[i][dataNames.index('residual sugar')]) for i in range(len(data))]
    chlorides = [float(data[i][dataNames.index('chlorides')]) for i in range(len(data))]
    free_sulfur_dioxide = [float(data[i][dataNames.index('free sulfur dioxide')]) for i in range(len(data))]
    total_sulfur_dioxide = [float(data[i][dataNames.index('total sulfur dioxide')]) for i in range(len(data))]
    density = [float(data[i][dataNames.index('density')]) for i in range(len(data))]
    pH = [float(data[i][dataNames.index('pH')]) for i in range(len(data))]
    sulphates = [float(data[i][dataNames.index('sulphates')]) for i in range(len(data))]
    alcohol = [float(data[i][dataNames.index('alcohol')]) for i in range(len(data))]
    return fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol
