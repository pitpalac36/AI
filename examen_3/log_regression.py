from math import exp


def sigmoid(x):
    return 1 / (1 + exp(-x))


def evaluate(xi, coef):
    yi = coef[0]
    for j in range(len(xi)):
        yi += coef[j + 1] * xi[j]
    return yi


class MyLogisticRegression:

    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []


    def fit(self, x, y, learningRate = 0.001, noEpochs = 10000):
        self.coef_ = [0.0 for _ in range(1 + len(x[0]))]
        # self.coef_ = [random.random() for _ in range(len(x[0]) + 1)]
        for epoch in range(noEpochs):
            for i in range(len(x)):
                ycomputed = sigmoid(evaluate(x[i], self.coef_))
                crtError = ycomputed - y[i]
                for j in range(0, len(x[0])):
                    self.coef_[j + 1] = self.coef_[j + 1] - learningRate * crtError * x[i][j]
                self.coef_[0] = self.coef_[0] - learningRate * crtError * 1
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]


    def predictOneSample(self, sampleFeatures):
        coefficients = [self.intercept_] + [c for c in self.coef_]
        if sigmoid(evaluate(sampleFeatures, coefficients)) < 0.5:
            return 'female'
        return 'male'
