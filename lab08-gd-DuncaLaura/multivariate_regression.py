import os
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from bgd_regressor import GDRegression
from file_utils import loadData2
from univariate_regression import check_liniarity


def plotOutputDataHistogram(x, xName):
    plt.hist(x, 10)
    plt.title('Histogram of ' + xName)
    plt.show()


def plotInputDataHistogram(x1, x2, x1Name, x2Name):
    plt.hist(x1, bins=10, alpha=0.5, label=x1Name)
    plt.hist(x2, bins=10, alpha=0.5, label=x2Name)
    plt.legend(loc='upper right')
    plt.title('Histogram of ' + x1Name + ' and ' + x2Name)
    plt.show()


def split_data2(input1, input2, output):
    numpy.random.seed(5)
    indexes = [i for i in range(len(input1))]
    trainSample = numpy.random.choice(indexes, int(0.8 * len(input1)), replace=False)
    testSample = [i for i in indexes if i not in trainSample]

    trainInputs = [[input1[i], input2[i]] for i in trainSample]
    trainOutputs = [output[i] for i in trainSample]

    testInputs = [[input1[i], input2[i]] for i in testSample]
    testOutputs = [output[i] for i in testSample]
    return trainInputs, trainOutputs, testInputs, testOutputs


def plot_test_and_train_data(trainInputs, trainOutputs, testInputs, testOutputs, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([trainInputs[i][0] for i in range(len(trainInputs))],
               [trainInputs[i][1] for i in range(len(trainInputs))], trainOutputs)
    ax.scatter([testInputs[i][0] for i in range(len(testInputs))],
               [testInputs[i][1] for i in range(len(testInputs))], testOutputs)
    ax.set_xlabel("GDP capita")
    ax.set_ylabel("Freedom")
    ax.set_zlabel("happiness")
    plt.title(title)
    plt.show()


def plot_learnt_model(trainInputs, trainOutputs, regr):
    xref = numpy.tile(numpy.arange(3), (3, 1))
    yref = xref.T
    zref = xref * regr.coef_[0] + yref * regr.coef_[1] + regr.intercept_
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter([trainInputs[i][0] for i in range(len(trainInputs))],
               [trainInputs[i][1] for i in range(len(trainInputs))], trainOutputs)
    ax.plot_surface(xref, yref, zref, alpha=0.3, color='yellow')
    ax.set_xlabel("GDP capita")
    ax.set_ylabel("Freedom")
    ax.set_zlabel("happiness")
    ax.set_title('Train data & learnt model')
    plt.show()


def plot_test_and_predicted_data(testInputs, testOutputs, regr):
    fig = plt.figure()
    ax = Axes3D(fig)
    predict = regr.predict(testInputs)
    ax.scatter([testInputs[i][0] for i in range(len(testInputs))],
               [testInputs[i][1] for i in range(len(testInputs))],
               testOutputs, zdir='z', s=20, c=None, depthshade=True)
    ax.scatter([testInputs[i][0] for i in range(len(testInputs))],
               [testInputs[i][1] for i in range(len(testInputs))],
               predict, zdir='z', s=20, c=None, depthshade=True)
    ax.set_title('Test Data & Predicted Data')
    ax.set_xlabel("GDP capita")
    ax.set_ylabel("Freedom")
    ax.set_zlabel("happiness")
    plt.show()
    return predict


def main(mode):
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'data', '2017.csv')
    economy_inputs, freedom_inputs, outputs = loadData2(filePath, 'Economy..GDP.per.Capita.', 'Freedom', 'Happiness.Score')
    plotInputDataHistogram(economy_inputs, freedom_inputs, 'capita GDP', 'Freedom')
    plotOutputDataHistogram(outputs, 'Happiness')
    check_liniarity(economy_inputs, outputs, 'GDP capita vs. happiness', 'GDP capita')
    check_liniarity(freedom_inputs, outputs, 'Freedom vs. happiness', 'Freedom')
    trainInputs, trainOutputs, testInputs, testOutputs = split_data2(economy_inputs, freedom_inputs, outputs)
    plot_test_and_train_data(trainInputs, trainOutputs, testInputs, testOutputs, "Test & train data")

    if mode == 'tool':
        regr = linear_model.SGDRegressor()
        for i in range(1000):
            regr.partial_fit(trainInputs, trainOutputs)
        w0, w1, w2 = regr.intercept_[0], regr.coef_[0], regr.coef_[1]
        print('Tool :')
        print('Intercept: w0 = ', w0)
        print('Coefficients: w1 = {}, w2 = {}'.format(w1, w2))
        plot_learnt_model(trainInputs, trainOutputs, regr)
        plot_test_and_predicted_data(testInputs, testOutputs, regr)
        print("prediction error : {}".format(mean_absolute_error(testOutputs, regr.predict(testInputs))))

    if mode == 'manual':
        regr = GDRegression()
        regr.fit(trainInputs, trainOutputs)
        w0, w1, w2 = regr.intercept_, regr.coef_[0], regr.coef_[1]
        print('Manual :')
        print('Intercept: w0 = ', w0)
        print('Coefficients: w1 = {}, w2 = {}'.format(w1, w2))
        plot_learnt_model(trainInputs, trainOutputs, regr)
        computedTestOutputs = plot_test_and_predicted_data(testInputs, testOutputs, regr)
        mae = sum(abs(r - c) for r, c in zip(testOutputs, computedTestOutputs)) / len(testOutputs)
        print("prediction error : {}".format(mae))
        # print("prediction error : {}".format(mean_absolute_error(testOutputs, regr.predict(testInputs))))


main('manual')
