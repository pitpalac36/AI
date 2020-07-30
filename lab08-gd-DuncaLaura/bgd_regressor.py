
class GDRegression:

    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    # batch GD
    def fit(self, x, y, learningRate=0.02, noEpochs=1000):
        self.coef_ = [0.0 for _ in range(len(x[0]) + 1)]
        # self.coef_ = [random.random() for _ in range(len(x[0]) + 1)]

        for _ in range(noEpochs):

            avg = [0.0 for _ in range(len(x[0]) + 1)]

            for i in range(len(x)):
                ycomputed = self.eval(x[i])
                crtError = ycomputed - y[i]

                for j in range(len(x[0])):
                    avg[j] += crtError * x[i][j]

                avg[len(x[0])] += crtError

            for k in range(len(x[0])):
                self.coef_[k] -= learningRate * avg[k] / len(x)

            self.coef_[len(x[0])] -= learningRate * avg[len(x[0])] / len(x)
            learningRate -= 1 / 100 * learningRate

        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]


    def eval(self, xi):
        yi = self.coef_[-1]
        for j in range(len(xi)):
            yi += self.coef_[j] * xi[j]
        return yi


    def predict(self, x):
        yComputed = [self.eval(xi) for xi in x]
        return yComputed
