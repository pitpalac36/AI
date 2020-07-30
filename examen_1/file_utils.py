import csv


def loadData(fileName, var1, var2):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariable = dataNames.index(var1)
    input1 = [float(data[i][selectedVariable]) for i in range(len(data))]  # pret
    selectedOutput = dataNames.index(var2)
    input2 = [int(data[i][selectedOutput]) for i in range(len(data))]  # zipcode
    return input1, input2


def loadData2(fileName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 151:
                break
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariable = dataNames.index('bedrooms')
    input1 = [int(data[i][selectedVariable]) for i in range(len(data))]  # bedrooms

    selectedOutput = dataNames.index('bathrooms')
    input2 = [float(data[i][selectedOutput]) for i in range(len(data))]  # bathrooms

    selectedOutput = dataNames.index('sqft_living')
    input3 = [int(data[i][selectedOutput]) for i in range(len(data))]  # sqft_living

    selectedOutput = dataNames.index('sqft_lot')
    input4 = [int(data[i][selectedOutput]) for i in range(len(data))]  # sqft_lot

    selectedOutput = dataNames.index('floors')
    input5 = [float(data[i][selectedOutput]) for i in range(len(data))]  # floors

    selectedOutput = dataNames.index('price')
    output = [int(data[i][selectedOutput]) for i in range(len(data))]  # prices

    return input1, input2, input3, input4, input5, output
