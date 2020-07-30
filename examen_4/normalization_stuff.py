from sklearn.preprocessing import StandardScaler


def normalisation(trainData, testData, una):
    scaler = StandardScaler()
    if not isinstance(trainData[0], list):
        # encode each sample into a list
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]
        unaData = [[d] for d in una]

        scaler.fit(trainData)  # fit only on training data
        normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data
        normUna = scaler.transform(unaData)

        # decode from list to raw values
        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
        normUna = [el[0] for el in normUna]
    else:
        scaler.fit(trainData)  # fit only on training data
        normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data
        normUna = scaler.transform(una)
    return normalisedTrainData, normalisedTestData, normUna


def normalization(data):
    minimum = min(data)
    maximum = max(data)
    normalized = []
    for each in data:
        each = (each - minimum) / (maximum - minimum)
        normalized.append(each)
    return normalized


def normalize_one(data, sample):
    minimum = []
    maximum = []
    for i in range(len(data[0])):
        mi = min([data[j][i] for j in range(len(data))])
        minimum.append(mi)
        ma = max([data[j][i] for j in range(len(data))])
        maximum.append(ma)
    for i in range(len(data[0])):
        sample[i] = (sample[i] - minimum[i]) / (maximum[i] - minimum[i])
    return sample
