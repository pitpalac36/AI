from math import sqrt


def regression_prediction(realOutputs, computedOutputs):
    mae = sum(abs(r - c) for r, c in zip(realOutputs, computedOutputs)) / len(realOutputs)
    rmse = sqrt(sum((r - c) ** 2 for r, c in zip(realOutputs, computedOutputs)) / len(realOutputs))
    return mae, rmse


def binary_labels_classification(realLabels, computedLabels, labelNames):
    accuracy = sum([1 if realLabels[i] == computedLabels[i] else 0 for i in range(len(realLabels))]) / len(realLabels)
    TP = sum(
        [1 if (realLabels[i] == labelNames['positive'] and computedLabels[i] == labelNames['positive']) else 0 for i in
         range(len(realLabels))])
    TN = sum(
        [1 if (realLabels[i] == labelNames['negative'] and computedLabels[i] == labelNames['negative']) else 0 for i in
         range(len(realLabels))])
    FP = sum(
        [1 if (realLabels[i] == labelNames['negative'] and computedLabels[i] == labelNames['positive']) else 0 for i in
         range(len(realLabels))])
    FN = sum(
        [1 if (realLabels[i] == labelNames['positive'] and computedLabels[i] == labelNames['negative']) else 0 for i in
         range(len(realLabels))])
    precisionPos = TP / (TP + FP)
    precisionNeg = TN / (TN + FN)
    recallPos = TP / (TP + FN)
    recallNeg = TN / (TN + FP)
    return accuracy, [precisionPos, precisionNeg], [recallPos, recallNeg]


def binary_probabilities_classification(realLabels, computedOutputs):
    computedLabels = []
    labelNames = list(set(realLabels))
    for each in computedOutputs:
        bigger_prob = each.index(max(each))
        label = labelNames[bigger_prob]
        computedLabels.append(label)
    labelNames = {'positive': labelNames[0], 'negative': labelNames[1]}
    return binary_labels_classification(realLabels, computedLabels, labelNames)


def multi_target_regression(realOutputs, computedOutputs):
    sae = 0
    quadratic_sum = 0
    k = len(realOutputs)
    noSamples = len(realOutputs[0])
    for i in range(k):
        for j in range(noSamples):
            sae += abs(realOutputs[i][j] - computedOutputs[i][j])
            quadratic_sum += (realOutputs[i][j] - computedOutputs[i][j]) ** 2
    mae = sae / (k * noSamples)
    rmse = sqrt(quadratic_sum / (k * noSamples))
    return mae, rmse


def multi_class_classification(realLabels, computedLabels):
    # computedLabels = []
    # for each in computedOutputs:
    # bigger_prob = each.index(max(each))
    # label = labelNames[bigger_prob]
    # computedLabels.append(label)
    labelNames = list(set(realLabels))
    accuracy = sum([1 if realLabels[i] == computedLabels[i] else 0 for i in range(len(realLabels))]) / len(realLabels)
    precision = {}
    recall = {}
    for each in labelNames:
        TP = FP = FN = 0
        for i in range(len(realLabels)):
            if realLabels[i] == each and computedLabels[i] == each:
                TP = TP + 1
            if realLabels[i] != each and computedLabels[i] == each:
                FP = FP + 1
            if realLabels[i] == each and computedLabels[i] != each:
                FN = FN + 1
        precision[each] = TP / (TP + FP)
        recall[each] = TP / (TP + FN)
    return accuracy, precision, recall
