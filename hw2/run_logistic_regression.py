from q2.check_grad import check_grad
from q2.utils import *
from q2.logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    # ====== Q2.2 (b) ======
    hyperparameters = {
        "learning_rate": 0.001,
        "weight_regularization": 0.,
        "num_iterations": 3000
    }
    weights = np.zeros((M+1, 1)) # weights with dimension (M + 1) x 1
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0. (should be less than 10^(-6))
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################

    # report hyperparameter settings you found worked the best
    print("=== best hyperparameters ===")
    for h in hyperparameters:
        print("{}: {}".format(h, hyperparameters[h]))
    print("{}: {}".format("initialized weights", "np.zeros((M+1, 1))"))

    # initializes shortcuts and arrays
    alpha = hyperparameters["learning_rate"]
    num_iterations = hyperparameters["num_iterations"]

    cross_entropies_T = np.ones(num_iterations)
    cross_entropies_V = np.ones(num_iterations)
    num_iter = np.linspace(1, num_iterations, num=num_iterations)

    # perform gradient descent
    for t in range(hyperparameters["num_iterations"]):
        f, df, y_train = logistic(weights,
                            train_inputs,
                            train_targets,
                            hyperparameters)
        weights = weights - alpha * df

        # record ce_train
        ce_train, frac_correct_train = evaluate(train_targets, y_train)
        cross_entropies_T[t] = ce_train[0][0]

        # record ce_valid
        y_valid = logistic_predict(weights, valid_inputs)
        ce_valid, frac_correct_valid = evaluate(valid_targets, y_valid)
        cross_entropies_V[t] = ce_valid[0][0]

    # report
    #   - the final cross entropy and
    #   - classification error
    #   on training, validation, and test sets

    print("== training == ")
    final_ce_T, frac_correct_T = evaluate(train_targets, y_train)
    print("final cross entropy (for training set): ", final_ce_T[0][0])
    print("Classification error (for training set): ", 1-frac_correct_T)

    print("== validation == ")
    final_ce_V, frac_correct_V = evaluate(valid_targets, y_valid)
    print("final cross entropy (for validation set): ", final_ce_V[0][0])
    print("Classification error (for validation set): ", 1-frac_correct_V)

    print("== test == ")
    test_inputs, test_targets = load_test()
    y_test = logistic_predict(weights, test_inputs)
    final_ce_test, frac_correct_test = evaluate(test_targets, y_test)
    print("final cross entropy (for test set): ", final_ce_test[0][0])
    print("Classification error (for test set): ", 1 - frac_correct_test)


    # ====== Q2.2 (c) ======
    # Plot graphs: for mnist_train (or mnist_train_small)
    # -- cross entropy vs. num of iterations
    plt.figure()
    plt.plot(num_iter, cross_entropies_T, '-r', label='training set')
    plt.plot(num_iter, cross_entropies_V, '-c', label='validation set')
    plt.xlabel('number of iterations')
    plt.ylabel('cross entropy')
    plt.grid()
    plt.legend()
    # plt.title(r"Q2.2(c-1) the cross entropy changes as "
    #           "training progresses for \'mnist_train\'", fontsize=10)
    # plt.savefig('Q2.2(c-1).png', dpi=100, quality=50)
    plt.title(r"Q2.2(c-2) the cross entropy changes as "
              "training progresses for \'mnist_train_small\'", fontsize=10)
    plt.savefig('Q2.2(c-2).png', dpi=100, quality=50)


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_pen_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Implement the function that automatically evaluates different     #
    # penalty and re-runs penalized logistic regression 5 times.        #
    #####################################################################
    # ====== Q2.3 (b) ======

    # hyperparameters for "mnist_train"
    hyperparameters = {
        "learning_rate": 0.1,
        "weight_regularization": 0.,
        "num_iterations": 2000
    }
    
    # hyperparameters for "mnist_train_small"
    # hyperparameters = {
    #     "learning_rate": 0.001,
    #     "weight_regularization": 0.,
    #     "num_iterations": 3000
    # }

    weight_regs = [0., 0.001, 0.01, 0.1, 1.0]
    alpha = hyperparameters["learning_rate"]
    num_iterations = hyperparameters["num_iterations"]


    for lamda in weight_regs:
        hyperparameters["weight_regularization"] = lamda

        # 1st col: train; 2nd col: validation
        ce_5_reruns = np.zeros((2, 5)) # (final) cross entropy for 5 re-runs
        cerror_5_reruns = np.zeros((2, 5)) # (final) cla. error for 5 re-runs

        for i in range(5):
            # initialize weights
            weights = np.zeros((M + 1, 1))  # weights with dimension (M + 1) x 1

            # initalize arrays
            cross_entropies_T = np.ones(num_iterations)
            cross_entropies_V = np.ones(num_iterations)
            num_iter = np.linspace(1, num_iterations, num=num_iterations)

            # perform gradient descent
            for t in range(hyperparameters["num_iterations"]):
                f, df, y_train = logistic_pen(weights,
                                          train_inputs,
                                          train_targets,
                                          hyperparameters)
                weights = weights - alpha * df

                # record ce_train (for each iteration)
                ce_train, frac_correct_train = evaluate(train_targets, y_train)
                cross_entropies_T[t] = ce_train[0][0]

                # record ce_valid (for each iteration)
                y_valid = logistic_predict(weights, valid_inputs)
                ce_valid, frac_correct_valid = evaluate(valid_targets, y_valid)
                cross_entropies_V[t] = ce_valid[0][0]


            # record final cross entropy and classification error
            # (for each rerun of given lamda)
            # print("== training == ")
            final_ce_T, frac_correct_T = evaluate(train_targets, y_train)
            final_ce_T, frac_correct_T = final_ce_T[0][0], 1 - frac_correct_T
            ce_5_reruns[0][i] = final_ce_T
            cerror_5_reruns[0][i] = frac_correct_T

            # print("== validation == ")
            final_ce_V, frac_correct_V = evaluate(valid_targets, y_valid)
            final_ce_V, frac_correct_V = final_ce_V[0][0], 1 - frac_correct_V
            ce_5_reruns[1][i] = final_ce_V
            cerror_5_reruns[1][i] = frac_correct_V



        # obtain and report
        #   - avg (final) cross entropy
        #   - avg (final) cla. error
        #   for training and validation
        avg_ce_T = np.average(ce_5_reruns[0])
        avg_ce_V = np.average(ce_5_reruns[1])
        avg_cerror_T = np.average(cerror_5_reruns[0])
        avg_cerror_V = np.average(cerror_5_reruns[1])

        print("====== for lamda={} ======".format(lamda))
        print("- Training")
        print("     Averaged (final) cross entropy: {}".format(avg_ce_T))
        print("     Averaged (final) cla. error: {}".format(avg_cerror_T))
        print("- Validation")
        print("     Averaged (final) cross entropy: {}".format(avg_ce_V))
        print("     Averaged (final) cla. error: {}".format(avg_cerror_V))




        # Plot graphs: for mnist_train (or mnist_train_small)
        # -- cross entropy vs. num of iterations
        # (with the LAST run of each lamda)
        plt.figure()
        plt.plot(num_iter, cross_entropies_T, '-r', label='training set')
        plt.plot(num_iter, cross_entropies_V, '-c', label='validation set')
        plt.xlabel('number of iterations')
        plt.ylabel('cross entropy')
        plt.grid()
        plt.legend()
        plt.title("Q2.3(b)(\'mnist_train\') the ce vs. num_iteration, "
                  "lamda={}".format(lamda), fontsize=9)
        plt.savefig('Q2.3(b-{}).png'.format(lamda), dpi=100, quality=50)
        # plt.title("Q2.3(b)(\'mnist_train_small\') the ce vs. num_iteration"
        #           ",lamda={}".format(lamda), fontsize=9)
        # plt.savefig('Q2.3(b-{}-small).png'.format(lamda), dpi=100, quality=50)




    # Q2.3 (c) test CE and classification with optimium lamda

    # perform gradient descent
    hyperparameters["weight_regularization"] = 0.1
    for t in range(hyperparameters["num_iterations"]):
        f, df, y_train = logistic_pen(weights,
                                      train_inputs,
                                      train_targets,
                                      hyperparameters)
        weights = weights - alpha * df
    print("== Q2.3 (c) test data == ")
    test_inputs, test_targets = load_test()
    y_test = logistic_predict(weights, test_inputs)
    final_ce_test, frac_correct_test = evaluate(test_targets, y_test)
    final_ce_test, frac_correct_test = \
        final_ce_test[0][0], 1 - frac_correct_test
    print("final cross entropy (for test set): ", final_ce_test)
    print("Classification error (for test set): ", frac_correct_test)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
    run_pen_logistic_regression()
