import os
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from bgd_regressor import GDRegression
from file_utils import loadData


def plotDataHistogram(x, variableName):
    plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()


def check_liniarity(inputs, outputs, title, independent_feature):
    plt.plot(inputs, outputs, 'ro')
    plt.xlabel(independent_feature)
    plt.ylabel('happiness')
    plt.title(title)
    plt.show()


def split_data(inputs, outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]
    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]
    return trainInputs, trainOutputs, testInputs, testOutputs


def plot_train_and_test(trainInputs, trainOutputs, testInputs, testOutputs, title):
    plt.plot(trainInputs, trainOutputs, 'ro', label='training data')
    plt.plot(testInputs, testOutputs, 'g^', label='testing data')
    plt.title(title)
    plt.xlabel('GDP capita')
    plt.ylabel('happiness')
    plt.legend()
    plt.show()


def train_model(trainInputs, trainOutputs):
    regr = linear_model.SGDRegressor()
    xx = [[el] for el in trainInputs]
    for i in range(1000):
        regr.partial_fit(xx, trainOutputs)
    return regr


def plot_model(trainInputs, trainOutputs, w0, w1, independent_feature, dependent_feature):
    noOfPoints = 1000
    xref = []
    val = min(trainInputs)
    step = (max(trainInputs) - min(trainInputs)) / noOfPoints
    for i in range(1, noOfPoints):
        xref.append(val)
        val += step
    yref = [w0 + w1 * el for el in xref]
    plt.plot(trainInputs, trainOutputs, 'ro', label='training data')
    plt.plot(xref, yref, 'b-', label='learnt model')
    plt.title('train data and the learnt model')
    plt.xlabel(independent_feature)
    plt.ylabel(dependent_feature)
    plt.legend()
    plt.show()


def predict_test(testInputs, testOutputs, regressor):
    computedTestOutputs = regressor.predict([[x] for x in testInputs])
    plt.plot(testInputs, computedTestOutputs, 'yo', label='computed test data')
    plt.plot(testInputs, testOutputs, 'g^', label='real test data')
    plt.title('computed test and real test data')
    plt.xlabel('GDP capita')
    plt.ylabel('happiness')
    plt.legend()
    plt.show()
    return computedTestOutputs


def main(mode):
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'data', '2017.csv')
    inputs, outputs = loadData(filePath, 'Economy..GDP.per.Capita.', 'Happiness.Score')
    plotDataHistogram(inputs, "capita GDP")
    plotDataHistogram(outputs, "Happiness score")
    check_liniarity(inputs, outputs, 'Capita vs happiness', 'GDP capita')
    trainInputs, trainOutputs, testInputs, testOutputs = split_data(inputs, outputs)
    plot_train_and_test(trainInputs, trainOutputs, testInputs, testOutputs, 'train and test data')
    xx = [[el] for el in trainInputs]

    if mode == 'tool':
        regressor = train_model(trainInputs, trainOutputs)
        w0, w1 = regressor.intercept_[0], regressor.coef_[0]
        print('Tool :')
        print('the learnt model : f(x) = ', w0, ' + ', w1, ' * x')
        plot_model(trainInputs, trainOutputs, w0, w1, 'GDP capita', 'happiness')
        computedTestOutputs = predict_test(testInputs, testOutputs, regressor)
        error = mean_squared_error(testOutputs, computedTestOutputs)
        print('prediction error :  ', error)

    if mode == 'manual':
        regressor = GDRegression()
        regressor.fit(xx, trainOutputs)
        w0, w1 = regressor.intercept_, regressor.coef_[0]
        print('Manual :')
        print('the learnt model : f(x) = ', w0, ' + ', w1, ' * x')
        plot_model(trainInputs, trainOutputs, w0, w1, 'GDP capita', 'happiness')
        computedTestOutputs = predict_test(testInputs, testOutputs, regressor)
        error = 0.0
        for t1, t2 in zip(computedTestOutputs, testOutputs):
            error += (t1 - t2) ** 2
        error = error / len(testOutputs)
        print("prediction error : ", error)


# main('tool')
