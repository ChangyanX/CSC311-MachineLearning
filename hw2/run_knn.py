from q2.l2_distance import l2_distance
from q2.utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # TODO:                                                             #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################

    # classification rate (on validation dataset):
    #       (num of correctly predicted cases) / (tot num of valid_inputs)
    classification_rate = []

    k_values = [1, 3, 5, 7, 9]
    for k in k_values:
        # get predicts of valid_inputs by knn
        valid_pred_k = knn(k, train_inputs, train_targets, valid_inputs)

        # calculate num of correct cases for this k
        count_correct = 0
        for i in range(len(valid_targets)):
            if valid_pred_k[i][0] == valid_targets[i][0]:
                count_correct += 1

        # compute the classification rate corresponding to this k
        rate = count_correct / len(valid_inputs)
        classification_rate.append(rate)

    # Plot the graph: classification rate of validation dataset vs. k value
    plt.figure()
    plt.plot(k_values, classification_rate, 'x-c')
    plt.xlabel('k')
    plt.ylabel('Classification Rate')
    plt.grid()
    plt.title('Q2.1(a) Classification Rate of Validation Dataset vs. k-value')
    plt.savefig('Q2.1(a).png')

    print("Q2.1(a) Classification rates:", classification_rate)



    # ======= Q2.1(b) =======
    chosen_k = 5        # 3 to 7
    k_values_chosen_k = [chosen_k-2, chosen_k, chosen_k+2]
    classification_rate_test = []

    for k in k_values_chosen_k:
        # get predicts of valid_inputs by knn
        test_pred_k = knn(k, train_inputs, train_targets, test_inputs)

        # calculate num of correct cases for this k
        count_correct = 0
        for i in range(len(test_targets)):
            if test_pred_k[i][0] == test_targets[i][0]:
                count_correct += 1

        # compute the classification rate corresponding to this k
        rate = count_correct / len(test_targets)
        classification_rate_test.append(rate)

    # Plot the graph: classification rate of validation dataset vs. k value
    plt.figure()
    plt.plot(k_values_chosen_k, classification_rate_test, 'x-g')
    plt.xlabel('k')
    plt.ylabel('Classification Rate')
    plt.grid()
    plt.title('Q2.1(b) Classification Rate of Test Dataset vs. k-value')
    plt.savefig('Q2.1(b).png')

    print("Q2.1(b) Classification rates:", classification_rate_test)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    run_knn()
