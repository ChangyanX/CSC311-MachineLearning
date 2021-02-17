from utils import *
from scipy.linalg import sqrtm

import numpy as np
import matplotlib.pyplot as plt


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


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


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################

    print("#############\n"
          "###  SVD  ###\n"
          "#############")

    # find best k with validation data using R from training data with given k
    # -- least reconstruction error
    # try at least 5 different k
    print("#3(a) find best k with validation data using R "
          "from training data with given k ")
    assumed_k = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200, 300, 400, 500]
    val_matrix = get_matrix_from_dict(val_data)

    rec_errs = []
    predict_acc = []
    for i, k in enumerate(assumed_k):
        R_svd = svd_reconstruct(train_matrix, k)
        diff = val_matrix - R_svd
        reconstruction_err = np.linalg.norm(diff[~np.isnan(diff)]) ** 2 \
                             / len(val_matrix)
        rec_errs.append(reconstruction_err)
        print("for k={}, \n"
              "reconstruction_err={}".format(k, reconstruction_err))

        # accuracy
        val_evaluation = sparse_matrix_evaluate(val_data, R_svd)
        predict_acc.append(val_evaluation)
        print("prediction accuracy: {}".format(val_evaluation))

    # find best k with validation set
    print("===== Chosen k =====")
    k1 = assumed_k[np.argmin(rec_errs)]
    k2 = assumed_k[np.argmax(predict_acc)]
    chosen_k = (k1+k2) // 2   # mean value of the lowest rec_err and highest acc
    print("chosen k = {}".format(chosen_k))


    # Report final validation and test perfomace with the chosen k
    print("===== Report final validation and test perfomace with the chosen k "
          "=====")
    R_svd = svd_reconstruct(train_matrix, chosen_k)


    print("-- validation set")
    diff = val_matrix - R_svd
    rec_err = np.linalg.norm(diff[~np.isnan(diff)])**2 / len(val_matrix)
    val_evaluation = sparse_matrix_evaluate(val_data, R_svd)
    print("for k={}, \n"
          "     reconstruction_err={}".format(chosen_k, rec_err))
    print("     prediction accuracy: {}".format(val_evaluation))



    print("-- test set")
    test_matrix = get_matrix_from_dict(test_data)
    diff = test_matrix - R_svd
    rec_err = np.linalg.norm(diff[~np.isnan(diff)]) ** 2 / len(test_matrix)
    test_evaluation = sparse_matrix_evaluate(test_data, R_svd)
    print("for k={}, \n"
          "     reconstruction_err={}".format(chosen_k, rec_err))
    print("     prediction accuracy: {}".format(test_evaluation))





    print("#3(b) Limitation of perform SVD: \n"
          "We basically treated the missing entries as the mean value of the \n"
          "possibility of correctness for each question according to all \n"
          "users. We only predict the with evaluation on question, but the \n"
          "correctness prediction on a single question for a single user is \n"
          "not evaluted horizontally, that is personally.\n")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                                #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    print("\n\n\n"
          "#############\n"
          "###  ALS  ###\n"
          "#############")

    print("#3(c) see functions 'als' and 'update_u_z'. ")
    print("#3(d) Learn U and Z with ALS. ")
    print("===== hyperparameters =====")
    lr = .1
    num_iter = 200000
    print("num_iterations = {}\n"
          "learning_rate = {}".format(num_iter, lr))

    print("===== train and find k* =====")
    assumed_k = [50, 100, 200, 300, 400, 500]
    als_matrix = []
    valid_evaluations = []
    for i, k in enumerate(assumed_k):
        train_matrix_als = als(train_data, k, lr=lr, num_iteration=num_iter)
        als_matrix.append(train_matrix_als)
        # accuracy
        valid_evaluation = sparse_matrix_evaluate(val_data, train_matrix_als)
        print("for k={}, prediction accuracy: {}".format(k, valid_evaluation))
        valid_evaluations.append(valid_evaluation)

    # with the minimum validation accuracy
    chosen_k = assumed_k[np.argmax(valid_evaluations)]
    print("chosen k = {}".format(chosen_k))




    # 3(e) plot square error loss vs. num_iter for training and validation data
    print("#3(e) plot with chosen k*. ")
    i = 500 # compute sq_err_loss every 500 iterations
    x = np.arange(0, num_iter, i)
    mat, train_errs, valid_errs, test_err = als(train_data, chosen_k, lr=lr,
                                                num_iteration=num_iter,
                                                plot=True,
                                                val_data=val_data,
                                                test_data=test_data)

    fig, axs = plt.subplots(2, sharex=True, sharey=False)
    fig.suptitle('Square Error Loss vs. Num iterations, '
                 'where k={}'.format(chosen_k))
    axs[0].plot(x, train_errs, 'tab:orange')
    axs[0].set_title('For training data')
    axs[1].plot(x, valid_errs, 'tab:green')
    axs[1].set_title('For validation data')

    for ax in axs.flat:
        ax.set(xlabel='num iterations', ylabel='square error loss')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig("#3(e) SqrErrLoss_vs_NumIter.png")

    # report how training and validation sq_err_loss change as a func of iter
    print("#3(e) report how training and validation sq_err_loss change as a "
          "func of iter")
    print("Ans: Both training and validation sq_err_loss first decrease \n"
          "exponentially, then converge to a constant range of loss as \n"
          "the number of iterations increases.")


    # report final validation accuracy and test accuracy
    print("#3(e) report final validation accuracy and test accuracy")
    valid_evaluation = sparse_matrix_evaluate(val_data, mat)
    test_evaluation = sparse_matrix_evaluate(test_data, mat)
    print("final validation accuracy: {}".format(valid_evaluation))
    print("final test accuracy: {}".format(test_evaluation))
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
