from utils import *
from scipy.linalg import sqrtm

import numpy as np
import matplotlib.pyplot as plt
import math

import time


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # perform stochastic gradient descent (SGD)
    k = len(z[0])
    u[n] = u[n] + lr * (np.array([[c]]) - u[n] @ z[q].T) @ z[q].reshape((1, k))
    z[q] = z[q] + lr * (np.array([[c]]) - u[n] @ z[q].T) @ u[n].reshape((1, k))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z

def als(train_data, k, lr, num_iteration, plot=False, val_data=None,
        test_data=None):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(max(set(train_data["user_id"]))+1, k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(max(set(train_data["question_id"]))+1, k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # initialize record
    per_i = 500     # record per_i num of iterations
    train_errs = np.zeros((num_iteration + 1) // per_i)
    valid_errs = np.zeros((num_iteration + 1) // per_i)

    for i in range(num_iteration):
        # update u, z
        u, z = update_u_z(train_data, lr, u, z)

        # 3(e) record squared_error_loss
        if plot and i % per_i == 0:
            train_err = squared_error_loss(train_data, u, z)
            train_errs[i // per_i] = train_err

            valid_err = squared_error_loss(val_data, u, z)
            valid_errs[i // per_i] = valid_err

    mat = u @ z.T

    if plot:
        test_err = squared_error_loss(test_data, u, z)
        return mat, train_errs, valid_errs, test_err
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat







# --------- NEW BELOW ----------

def als_updated(train_data, k, lr0, drop, epochs_drop, num_iteration,
                mu, lamda_uf, lamda_ub, lamda_zf, lamda_zb,
                plot=False, val_data=None, test_data=None):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    n = max(set(train_data["user_id"])) + 1
    m = max(set(train_data["question_id"])) + 1

    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(max(set(train_data["user_id"]))+1, k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(max(set(train_data["question_id"]))+1, k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    train_mat = get_matrix_from_dict(train_data)
    # mu = np.mean(train_mat[~np.isnan(train_mat)])                # global_bias
    user_bias = np.zeros(max(set(train_data["user_id"]))+1)      # user_bias
    ques_bias = np.zeros(max(set(train_data["question_id"]))+1)  # question_bias

    # initialize record
    per_i = 500     # record per_i num of iterations
    train_errs = np.zeros((num_iteration + 1) // per_i)
    valid_errs = np.zeros((num_iteration + 1) // per_i)

    for i in range(num_iteration):
        # update u, z
        lr = update_lr(lr0, drop, epochs_drop, i)
        # lr = lr0
        u, z , user_bias, ques_bias = \
            update_u_z_new(train_data, lr, u, z, mu, user_bias, ques_bias,
                           lamda_uf, lamda_ub, lamda_zf, lamda_zb)

        # 3(e) record squared_error_loss
        if plot and i % per_i == 0:
            train_err = squared_error_loss(train_data, u, z)
            train_errs[i // per_i] = train_err

            valid_err = squared_error_loss(val_data, u, z)
            valid_errs[i // per_i] = valid_err


    global_bias_mat = np.ones((n, m)) * mu
    user_bias_mat = user_bias.reshape((n, 1)) @ np.ones((1, m))
    ques_bias_mat = np.ones((n, 1)) @ ques_bias.reshape((1, m))

    mat = global_bias_mat + user_bias_mat + ques_bias_mat + u @ z.T

    if plot:
        test_err = squared_error_loss(test_data, u, z)
        return mat, train_errs, valid_errs, test_err
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def update_u_z_new(train_data, lr, u, z, mu, user_bias, ques_bias,
                   lamda_uf, lamda_ub, lamda_zf, lamda_zb):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z, bu, bz)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # make predictions and obtain error
    predict = prediction(mu, user_bias, ques_bias, u, z, n, q) # \hat{C}_{nm}
    err = np.array([[c]]) - predict

    # perform stochastic gradient descent (SGD)
    k = len(z[0])
    user_bias[n] = user_bias[n] + lr * (err - lamda_ub * user_bias[n])
    ques_bias[q] = ques_bias[q] + lr * (err - lamda_zb * ques_bias[q])
    u[n] = u[n] + lr * (err @ z[q].reshape((1, k)) - lamda_uf * u[n])
    z[q] = z[q] + lr * (err @ u[n].reshape((1, k)) - lamda_zf * z[q])

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z, user_bias, ques_bias



def prediction(mu, user_bias, question_bias, u, z, n, m):
    return mu + user_bias[n] + question_bias[m] + u[n] @ z[m].T


def update_lr(lr0, drop, epochs_drop, iter_num):
    """
    Return the new learning rate <lrate> after update.

    learning rate should update
     - step decay and converge to 0.01
    """
    lrate = lr0 * math.pow(drop, math.floor((1 + iter_num) / epochs_drop))+0.01
    return lrate


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################

    print("#############\n"
          "###  ALS  ###\n"
          "#############")

    # Compare the runtime, final accuracy -- time-based decay for the learning
    # rate
    print("\n\n\n"
          "Updated ALS with L2 Regularization and step-decay learning rate")

    print("===== hyperparameters =====")
    num_iter = 145000
    chosen_k = 380
    lr0 = 0.3 # or 0.1 for constant learning rate

    lamda_uf, lamda_ub, lamda_zf, lamda_zb = .4, 0.15, 0.15, .3
    mu = 0.5

    drop = 0.15
    epochs_drop = num_iter // 5
    print("num_iterations = {}\n"
          "inital learning_rate = {}\n"
          "chosen_k = {}\n"
          "global bias = {}\n"
          "user bias reg = {}\n"
          "question bias reg = {}\n"
          "user factor reg = {}\n"
          "question factor reg = {}\n"
          "drop = {}\n"
          "epochs_drop = {}\n"
          "".format(num_iter, lr0, chosen_k, mu,
                    lamda_ub, lamda_zb, lamda_uf, lamda_zf,
                    drop, epochs_drop))


    # # use the follow to tune hyperparameters for Updated ALS with L2 Regularization
    # iters = [2000, 3000, 3100, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
    #          11000, 12000, 13000, 14000, 15000]
    # for num_iter in iters:
    #     print("num_iter=", num_iter)
    #
    #     als_start = time.perf_counter()
    #     mat = als_updated(train_data, k=chosen_k, lr0=lr0, drop=drop,
    #                       epochs_drop=epochs_drop, num_iteration=num_iter,
    #                       mu=mu,
    #                       lamda_uf=lamda_uf, lamda_ub=lamda_ub,
    #                       lamda_zf=lamda_zf, lamda_zb=lamda_zb)
    #     als_end = time.perf_counter()
    #     print("     als_start={}, "
    #           "als_end={}, "
    #           "als_duration={}".format(als_start, als_end, als_end - als_start))
    #     valid_evaluation = sparse_matrix_evaluate(val_data, mat)
    #     test_evaluation = sparse_matrix_evaluate(test_data, mat)
    #     print("     final validation accuracy: {}".format(valid_evaluation))
    #     print("     final test accuracy: {}".format(test_evaluation))


    als_start = time.perf_counter()
    mat = als_updated(train_data, k=chosen_k, lr0=lr0, drop=drop,
                      epochs_drop=epochs_drop, num_iteration=num_iter, mu=mu,
                      lamda_uf=lamda_uf, lamda_ub=lamda_ub,
                      lamda_zf=lamda_zf, lamda_zb=lamda_zb)
    als_end = time.perf_counter()
    print("     als_start={}, "
          "als_end={}, "
          "als_duration={}".format(als_start, als_end, als_end - als_start))

    # report final validation accuracy and test accuracy
    valid_evaluation = sparse_matrix_evaluate(val_data, mat)
    test_evaluation = sparse_matrix_evaluate(test_data, mat)
    print("     final validation accuracy: {}".format(valid_evaluation))
    print("     final test accuracy: {}".format(test_evaluation))




    print("\n\n\n"
          "Original ALS")
    print("===== hyperparameters =====")
    lr = .1  # orginal constant learning rate
    num_iter = 200000
    chosen_k = 200
    print("num_iterations = {}\n"
          "constant learning_rate = {}\n"
          "chosen_k = {}\n".format(num_iter, lr, chosen_k))


    als_start = time.perf_counter()
    mat0 = als(train_data, chosen_k, lr, num_iter)
    als_end = time.perf_counter()
    print("     als_start={}, "
          "als_end={}, "
          "als_duration={}".format(als_start, als_end, als_end - als_start))

    # report final validation accuracy and test accuracy
    valid_evaluation = sparse_matrix_evaluate(val_data, mat0)
    test_evaluation = sparse_matrix_evaluate(test_data, mat0)
    print("     final validation accuracy: {}".format(valid_evaluation))
    print("     final test accuracy: {}".format(test_evaluation))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def get_matrix_from_dict(data):
    """ Return a 2D matrix of 542x1774, where unobserved entries are given 'NaN'

    :param data: A dictionary {user_id: list, question_id: list, is_correct:
    list}
    :return: 2D Matrix.
    """
    matrix = np.ones((542, 1774)) * np.nan
    for i, ic in enumerate(data["is_correct"]):
        qid = data["question_id"][i]
        uid = data["user_id"][i]
        matrix[uid][qid] = ic

    return matrix

if __name__ == "__main__":
    main()
