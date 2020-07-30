from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from logistic_regression import MyLogisticRegression
from normalization import tool_normalisation


def plotDataHistogram(x, variableName):
    plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
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


def main(mode):
    data = load_iris()
    inputs = data['data']
    outputs = data['target']
    outputNames = data['target_names']
    featureNames = list(data['feature_names'])

    feature1 = [feat[featureNames.index('sepal length (cm)')] for feat in inputs]
    feature2 = [feat[featureNames.index('sepal width (cm)')] for feat in inputs]
    feature3 = [feat[featureNames.index('petal length (cm)')] for feat in inputs]
    feature4 = [feat[featureNames.index('petal width (cm)')] for feat in inputs]

    inputs = [[feat[featureNames.index('sepal length (cm)')],
               feat[featureNames.index('sepal width (cm)')],
               feat[featureNames.index('petal length (cm)')],
               feat[featureNames.index('petal width (cm)')]] for feat in inputs]

    plotDataHistogram(feature1, 'sepal length (cm)')
    plotDataHistogram(feature2, 'sepal width (cm)')
    plotDataHistogram(feature3, 'petal length (cm)')
    plotDataHistogram(feature4, 'petal width (cm)')
    plotDataHistogram(outputs, 'setosa, versicolor & virginica')

    trainInputs, trainOutputs, testInputs, testOutputs = split_data(inputs, outputs)

    if mode == 'tool':
        trainInputs, testInputs = tool_normalisation(trainInputs, testInputs)
        classifier = linear_model.LogisticRegression()
        classifier.fit(trainInputs, trainOutputs)
        for i in range(len(outputNames)):
            w0, w1, w2, w3, w4 = classifier.intercept_[i], classifier.coef_[i][0], classifier.coef_[i][1], \
                                 classifier.coef_[i][2], classifier.coef_[i][3]
            print('classification model ' + outputNames[i] + ' : y(feat1, feat2, feat3, feat4) = ', w0, ' + ', w1, ' * feat1 + ', w2,
                  ' * feat2 + ', w3, ' * feat3 + ', w4, ' * feat4')
        computedTestOutputs = classifier.predict(testInputs)
        error = 1 - accuracy_score(testOutputs, computedTestOutputs)
        print("\nclassification error : {}\n".format(error))

    if mode == 'manual':
        trainOutputsSetosa = [1 if trainOutputs[i] == 0 else 0 for i in range(len(trainOutputs))]
        trainOutputsVersicolor = [1 if trainOutputs[i] == 1 else 0 for i in range(len(trainOutputs))]
        trainOutputsVirginica = [1 if trainOutputs[i] == 2 else 0 for i in range(len(trainOutputs))]

        classifierSetosa = MyLogisticRegression()
        classifierSetosa.fit(trainInputs, trainOutputsSetosa)

        classifierVersicolor = MyLogisticRegression()
        classifierVersicolor.fit(trainInputs, trainOutputsVersicolor)

        classifierVirginica = MyLogisticRegression()
        classifierVirginica.fit(trainInputs, trainOutputsVirginica)

        for i in range(len(outputNames)):
            if outputNames[i] == 'setosa':
                classifier = classifierSetosa
            elif outputNames[i] == 'versicolor':
                classifier = classifierVersicolor
            else:
                classifier = classifierVirginica
            w0, w1, w2, w3, w4 = classifier.intercept_, classifier.coef_[0], classifier.coef_[1], \
                                 classifier.coef_[2], classifier.coef_[3]
            print('classification model ' + outputNames[i] + ' : y(feat1, feat2, feat3, feat4) = ', w0, ' + ', w1, ' * feat1 + ', w2,
                  ' * feat2 + ', w3, ' * feat3 + ', w4, ' * feat4')

        accuracy = 0
        for i in range(len(testInputs)):
            # setosa -> 0, versicolor -> 1 & virginica -> 2
            dictionary = {'setosa': classifierSetosa.predictOneSample(testInputs[i]),
                          'versicolor': classifierVersicolor.predictOneSample(testInputs[i]),
                          'virginica': classifierVirginica.predictOneSample(testInputs[i])}

            computed = max(dictionary, key=dictionary.get)
            real = 'setosa' if testOutputs[i] == 0 else 'versicolor' if testOutputs[i] == 1 else 'virginica'
            if real == computed:
                accuracy += 1
                print("computed : " + computed + "     real : " + real)
            else:
                print("computed : " + computed + "     real : " + real + "   GRESIT")
        error = 1 - (accuracy / len(testOutputs))
        print("error : " + str(error))


main('manual')
