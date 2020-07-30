class ManualRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef1_ = 0.0
        self.coef2_ = 0.0


    # x - input (list of economy & freedom pairs)
    # y - output
    def fit(self, x, y):
        first_feature_list = [x[i][0] for i in range(len(x))]
        second_feature_list = [x[i][1] for i in range(len(x))]

        first_sum = sum(i * j for (i, j) in zip(first_feature_list, y))
        second_sum = sum(i * j for (i, j) in zip(second_feature_list, y))
        feature_prod_sum = sum(i * j for (i, j) in zip(first_feature_list, second_feature_list))
        sum_f1 = len(first_feature_list) * sum(first_feature_list[i] ** 2 for i in range(len(first_feature_list))) \
                 - sum(first_feature_list[i] for i in range(len(first_feature_list))) ** 2
        sum_f2 = len(second_feature_list) * sum([second_feature_list[i] ** 2 for i in range(len(second_feature_list))]) - sum(second_feature_list) ** 2
        another_sum = len(first_feature_list) * feature_prod_sum - sum(first_feature_list) * sum(second_feature_list)
        sum_output1 = len(second_feature_list) * second_sum - sum(second_feature_list) * sum(y)
        sum_output2 = len(first_feature_list) * first_sum - sum(first_feature_list) * sum(y)

        self.coef1_ = (sum_f2 * sum_output2 - another_sum * sum_output1) / (sum_f1 * sum_f2 - another_sum ** 2)
        self.coef2_ = (sum_f1 * sum_output1 - another_sum * sum_output2) / (sum_f1 * sum_f2 - another_sum ** 2)
        self.intercept_ = (sum(y) - self.coef1_ * sum(first_feature_list) - self.coef2_ * sum(second_feature_list)) / len(first_feature_list)
        return self.intercept_, self.coef1_, self.coef2_

    def predict(self, x):
        return [self.intercept_ + self.coef1_ * val[0] + self.coef2_ * val[1] for val in x]
