import random
import os
from file_utils import loadData
from tool_main import plotInputDataHistogram, plotOutputDataHistogram, check_liniarity, plot_test_and_train_data


def run_manual():
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
    random.seed(100)
    indexes = [i for i in range(len(economy_inputs))]
    trainSample = random.sample(indexes, int(0.8 * len(economy_inputs)))
    testSample = [i for i in indexes if i not in trainSample]
    trainInputs = [[economy_inputs[i], freedom_inputs[i]] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [[economy_inputs[i], freedom_inputs[i]] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    # plot test and train data
    plot_test_and_train_data(trainInputs, trainOutputs, testInputs, testOutputs)


run_manual()
