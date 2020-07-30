import numpy as np
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from feature_extraction import read_from_csv, tokenize_sentences, bag_of_words
from knn import kmeans, euclidean_distance, jaccard_similarity
from log_regression import MyLogisticRegression


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



def data2FeaturesMoreClasses(inputs, outputs, outputNames):
    labels = set(outputs)
    noData = len(inputs)
    for crtLabel in labels:
        x = [inputs[i][0] for i in range(noData) if outputs[i] == crtLabel]
        y = [inputs[i][1] for i in range(noData) if outputs[i] == crtLabel]
        plt.scatter(x, y, label=outputNames[crtLabel])
    plt.xlabel('feat1')
    plt.ylabel('feat2')
    plt.legend()
    plt.show()



def iris_classification():
    data = load_iris()
    inputs = data['data']
    outputs = data['target']
    featureNames = list(data['feature_names'])
    outputNames = ['setosa', 'versicolor', 'virginica']

    inputs = [[feat[featureNames.index('sepal length (cm)')],
               feat[featureNames.index('sepal width (cm)')],
               feat[featureNames.index('petal length (cm)')],
               feat[featureNames.index('petal width (cm)')]] for feat in inputs]

    trainInputs, trainOutputs, testInputs, testOutputs = split_data(inputs, outputs)
    data2FeaturesMoreClasses(trainInputs, trainOutputs, outputNames)

    km = kmeans(k=3, similarity=euclidean_distance, max_iterations=50)
    km.fit(trainInputs)

    print(len(km.clusters[0]))
    print(len(km.clusters[1]))
    print(len(km.clusters[2]))

    for each in range(len(km.centroids)):
        plt.scatter(km.centroids[each][0], km.centroids[each][1], s=130, marker="*")
    data2FeaturesMoreClasses(trainInputs, trainOutputs, outputNames)

    for i in range(len(testInputs)):
        print('real : ' + str(testOutputs[i]) + '   predicted : ' + str(km.predictEuclidean(testInputs[i])))
    print('Dunn index : ' + str(km.dunn_index()))



def text_classification(mode):
    words_inputs, word_outputs = read_from_csv('reviews_mixed.csv')
    vocabulary = tokenize_sentences(words_inputs)
    bag = []
    for each in words_inputs:
        bag.append(bag_of_words(each, vocabulary))
    bag_train_input, bag_train_output, bag_test_input, bag_test_output = split_data(bag, word_outputs)

    if mode == 'kmeans':
        words_km = kmeans(k=2, similarity=jaccard_similarity, max_iterations=50)
        words_km.fit(bag_train_input)

        print(len(words_km.clusters[0]))
        print(len(words_km.clusters[1]))
        dictionary = {'first cluster' : [], 'second cluster' : []}
        for i in range(len(bag_test_input)):
            if words_km.centroids[0] == words_km.predictJaccard(bag_test_input[i]):
                dictionary['first cluster'].append(bag_test_output[i])
            else:
                dictionary['second cluster'].append(bag_test_output[i])
        print('FIRST CLUSTER :')
        for each in dictionary['first cluster']:
            print(each)
        print('SECOND CLUSTER :')
        for each in dictionary['second cluster']:
            print(each)
        print('Dunn index : ' + str(words_km.dunn_index()))


    if mode == 'log':
        classifier = MyLogisticRegression()
        numeric_output = [0 if bag_train_output[i] == 'negative' else 1 for i in range(len(bag_train_output))]
        classifier.fit(bag_train_input, numeric_output)
        accuracy = 0
        for i in range(len(bag_test_input)):
            if classifier.predictOneSample(bag_test_input[i]) > 0.5:
                computed = 'positive'
            else:
                computed = 'negative'
            real = bag_train_output[i]
            if real == computed:
                accuracy += 1
                print("computed : " + computed + "     real : " + real)
            else:
                print("computed : " + computed + "     real : " + real + "   GRESIT")
        error = 1 - (accuracy / len(bag_test_input))
        print("error : " + str(error))


# iris_classification()
text_classification('log')
