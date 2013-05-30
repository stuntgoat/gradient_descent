from math import exp, log
from gradient_descent import gradient_descent, scale_add, DiffFunc
"""
Given a loss function we are going to calculate the classifier's weights using
gradient descent.
"""


def feature_function(x, y):

    if y == '+':
        return RAW[x] + [0, 0]
    elif y == '-':
        return  [0, 0] + RAW[x]

class LogisticRegressionLoss(DiffFunc):

    def __init__(self, feature_func, dataset=[]):
        self.feature_func = feature_func
        self.possible_y_values = set([x[1] for x in dataset])
        self.dataset = dataset

    def f(self, weights=[]):
        # datum is (x, y); where x is the feature vector and y is the label.
        return -(sum([log(self.p_of_y_given_x(weights, datum[1], datum[0])) for datum in self.dataset]))

    def diff_of_f(self, weights=[]):
        total = [0 for x in weights]

        for datum in self.dataset:
            total = scale_add(total, 1, self.feature_func(datum[0], datum[1]))
            for y in self.possible_y_values:
                p = self.p_of_y_given_x(weights, y, datum[0])
                total = scale_add(total, -p, self.feature_func(datum[0], y))

        return [-(x) for x in total]

    def p_of_y_given_x(self, weights, y, x):
        numerator = exp(self.dot_product(weights, self.feature_func(x, y)))

        denominator = sum([exp(self.dot_product(weights, self.feature_func(x, y1))) for y1 in self.possible_y_values])

        return numerator / denominator

    def dot_product(self, weights, features):
        return sum([x * y for x, y in zip(weights, features)])

    def predict(self, x, weights):
        calculated_probs = []

        for y in self.possible_y_values:
            calculated_probs.append((self.p_of_y_given_x(weights, y, x), y))

        assert abs(sum([d[0] for d in calculated_probs]) - 1) < 1e-8

        return max(calculated_probs)[1]


###############################################################################
# Notes:

example_dataset = [(1, '+'), (2, '+'), (3, '+'), (4, '+'), (5, '-'), (6, '-'), (7, '-')]

review_loss_func = LogisticRegressionLoss(feature_function, example_dataset)

weights = gradient_descent([10, 10, 10, 10], .3, review_loss_func)

# for d in test_data:
#     prediction = review_loss_func.predict(d, weights)
#     print d, RAW[d], review_loss_func.p_of_y_given_x(weights, prediction, d), prediction



# print review_loss_func.predict(8, weights)


# for d in example_dataset:
#     print d, review_loss_func.predict(d[0], weights)


    # print d, review_loss_func.p_of_y_given_x(weights, d[1], d[0])

# iris_loss = LogisticRegressionLoss()


# weights = gradient_descent(guess, alpha, iris_loss)

# RAW = {
#     1: [2, 2],
#     2: [2, 4],
#     3: [3, 3],
#     4: [3, 5],
#     5: [-4, -1.5],
#     6: [-2, -2],
#     7: [-2, -1],
#     8: [3, 2],
#     9: [10, 10],
#     10: [-3, 3],
#     11: [3, -3],
#     12: [-5, -5],
#     13: [5, 5],
#     14: [-3, 0.01],
#     15: [2, 0.1],
#     }



# training_data = [(1, '+'), (2, '+'), (3, '+'), (4, '+'), (5, '-'), (6, '-'), (7, '-')]

# # test_data = [8, 9, 10, 11, 12, 13]

# test_data = [14, 15]
