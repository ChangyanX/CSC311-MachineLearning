from q2.utils import sigmoid

import numpy as np


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # TODO:                                                             #
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################
    dummy = np.ones((len(data), 1))
    x = np.hstack((data, dummy))  # N X (M+1), last element -> bias
    w = weights  # (M+1) x 1
    z = np.dot(x, w)  # z = xw

    y = sigmoid(z)  # y = sigma(z)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO:                                                             #
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################
    # find ce
    ones_len_t = np.ones((len(targets), 1))
    ones_len_y = np.ones((len(y), 1))
    N = len(targets)

    ce_tot = - np.dot(targets.T, np.log(y)) \
             - np.dot((ones_len_t - targets).T, np.log(ones_len_y - y))
    ce = ce_tot / N

    # find frac_correct
    correct = 0
    for i in range(len(targets)):
        # threshold for logistic =1/2
        if (targets[i][0] == 1. and y[i][0] >= 0.5) or \
                (targets[i][0] == 0. and y[i][0] <= 0.5):
            correct += 1
    frac_correct = correct / N
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the same as averaged cross entropy.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of parameters, and the probabilities given by    #
    # logistic regression.                                              #
    #####################################################################
    # Hint: hyperparameters will not be used here.

    N = len(data)
    ce, frac_correct = evaluate(targets, y)

    f = ce
    dummy = np.ones((len(data), 1))
    x = np.hstack((data, dummy))  # N X (M+1), last element -> bias
    df = np.dot(x.T, (y - targets)) / N  # df/dw = (X.T)•(y-t) /N


    # ======test=====
    # print("weights shape: ", weights.shape)
    # print("f shape: ", f.shape)
    # print("df shape: ", df.shape)
    # print("y shape: ", y.shape)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """ Calculate the cost of penalized logistic regression and its derivatives
    with respect to weights. Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points, plus a penalty term.
           This is the objective that we want to minimize.
        df: (M+1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points (plus a penalty term), gradient of parameters, and the     #
    # probabilities given by penalized logistic regression.             #
    #####################################################################

    # find penalized f
    N = len(data)
    ce, frac_correct = evaluate(targets, y)
    lamda = hyperparameters["weight_regularization"]
    no_bias_weights = np.copy(weights)
    no_bias_weights[-1] = 0
    f = ce + lamda * (1/2) * np.dot(no_bias_weights.T, no_bias_weights)

    # find penalized df
    dummy = np.ones((len(data), 1))
    x = np.hstack((data, dummy))  # N X (M+1), last element -> bias
    df = np.dot(x.T, (y - targets)) / N + lamda * no_bias_weights
    # df/dw = (X.T)•(y-t) /N + lamda * w', where w' has the bias (last) term
    # to be 0

    # ======test=====
    # print("weights shape: ", weights.shape)
    # print("f shape: ", f.shape)
    # print("df shape: ", df.shape)
    # print("y shape: ", y.shape)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y
