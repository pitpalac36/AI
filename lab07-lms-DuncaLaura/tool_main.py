import os
import matplotlib.pyplot as plt
import numpy
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from file_utils import loadData


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


def check_liniarity(input, output, titlu, dependent_variable_label):
    plt.plot(input, output, 'ro')
    plt.xlabel(dependent_variable_label)
    plt.ylabel('happiness')
    plt.title(titlu)
    plt.show()


def split_data(input1, input2, output):
    numpy.random.seed(5)
    indexes = [i for i in range(len(input1))]
    trainSample = numpy.random.choice(indexes, int(0.8 * len(input1)), replace=False)
    testSample = [i for i in indexes if i not in trainSample]

    trainInputs = [[input1[i], input2[i]] for i in trainSample]
    trainOutputs = [output[i] for i in trainSample]

    testInputs = [[input1[i], input2[i]] for i in testSample]
    testOutputs = [output[i] for i in testSample]
    return trainInputs, trainOutputs, testInputs, testOutputs


def plot_test_and_train_data(trainInputs, trainOutputs, testInputs, testOutputs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([trainInputs[i][0] for i in range(len(trainInputs))],
               [trainInputs[i][1] for i in range(len(trainInputs))], trainOutputs)
    ax.scatter([testInputs[i][0] for i in range(len(testInputs))],
               [testInputs[i][1] for i in range(len(testInputs))], testOutputs)
    ax.set_xlabel("GDP capita")
    ax.set_ylabel("Freedom")
    ax.set_zlabel("happiness")
    plt.title("Test & train data")
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


def predict_sklearn(trainInputs, trainOutputs, regr):
    xref = numpy.tile(numpy.arange(3), (3, 1))
    yref = xref.T
    zref = xref * regr.coef_[0] + yref * regr.coef_[1] + regr.intercept_
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter([trainInputs[i][0] for i in range(len(trainInputs))], [trainInputs[i][1] for i in range(len(trainInputs))], trainOutputs)
    ax.plot_surface(xref, yref, zref, alpha=0.3, color='yellow')
    ax.set_xlabel("GDP capita")
    ax.set_ylabel("Freedom")
    ax.set_zlabel("happiness")
    ax.set_title('Train data & learnt model')
    plt.show()


def run_sklearn():
    # load data from csv
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'data', '2017.csv')

    economy_inputs, freedom_inputs, outputs = loadData(filePath, 'Economy..GDP.per.Capita.', 'Freedom', 'Happiness.Score')

    # plot data
    plotInputDataHistogram(economy_inputs, freedom_inputs, 'capita GDP', 'Freedom')  # plot data
    plotOutputDataHistogram(outputs, 'Happiness')

    # check liniarity
    check_liniarity(economy_inputs, outputs, 'GDP capita vs. happiness', 'GDP capita')
    check_liniarity(freedom_inputs, outputs, 'Freedom vs. happiness', 'Freedom')

    # split data
    trainInputs, trainOutputs, testInputs, testOutputs = split_data(economy_inputs, freedom_inputs, outputs)

    # plot test and train data
    plot_test_and_train_data(trainInputs, trainOutputs, testInputs, testOutputs)

    # dictionary = {'GDP capita': economy_inputs, 'Freedom': freedom_inputs, 'happiness': outputs}
    # df = pandas.DataFrame(dictionary, columns=['GDP capita', 'Freedom', 'happiness'])
    # trainInputs = df[['GDP capita', 'Freedom']]
    # trainOutputs = df['happiness']

    # regressor with sklearn
    regr = linear_model.LinearRegression()
    regr.fit(trainInputs, trainOutputs)
    w0, w1, w2 = regr.intercept_, regr.coef_[0], regr.coef_[1]
    print('Intercept: w0 = ', w0)
    print('Coefficients: w1 = {}, w2 = {}'.format(w1, w2))

    # plot the learnt model
    predict_sklearn(trainInputs, trainOutputs, regr)

    # plot test and predicted data
    plot_test_and_predicted_data(testInputs, testOutputs, regr)

    print("Mean absolute Error : {}".format(mean_absolute_error(trainOutputs, regr.predict(trainInputs))))


# run_sklearn()
