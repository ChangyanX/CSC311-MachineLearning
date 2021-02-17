from utils import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = np.zeros((len(theta), len(beta)))
    for i, y in enumerate(data["is_correct"]):
        u = data["user_id"][i]
        v = data["question_id"][i]
        x = (theta[u] - beta[v]).sum()
        p = sigmoid(x)
        log_lklihood[u, v] = y * np.log(p) + (1-y) * np.log(1-p)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for i, y in enumerate(data["is_correct"]):
        u = data["user_id"][i]
        v = data["question_id"][i]
        x = (theta[u] - beta[v]).sum()
        p = sigmoid(x)
        theta[u] = theta[u] + lr * (y-p)
        beta[v] = beta[v] - lr * (y-p)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.rand(max(data['user_id'])+1)
    beta = np.random.rand(max(data['question_id'])+1)

    val_acc_lst = []
    train_neg_lld_list = []
    val_neg_lld_list = []
    for i in range(iterations):
        print(i)
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_neg_lld_list.append(np.sum(neg_lld))
        val_neg_lld_list.append(np.sum(neg_log_likelihood(val_data, theta=theta, beta=beta)))
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        #print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_neg_lld_list, val_neg_lld_list


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    iter = 30
    theta, beta, val_acc_lst,  train_neg_lld_list, val_neg_lld_list = irt(train_data, val_data, lr, iter)

    plt.figure(figsize=(10, 8))
    plt.plot([str(k) for k in range(len(train_neg_lld_list))], train_neg_lld_list)
    plt.xlabel("iteration")
    plt.ylabel("log-likelihoods")
    plt.title("log-likelihoods on the train data")
    plt.savefig("#2(b) train.png")

    plt.figure(figsize=(10, 8))
    plt.plot([str(k) for k in range(len(val_neg_lld_list))], val_neg_lld_list)
    plt.xlabel("iteration")
    plt.ylabel("log-likelihoods")
    plt.title("log-likelihoods on the validation data")
    plt.savefig("#2(b) validation.png")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)
    plt.figure(figsize=(10, 8))
    plt.plot([str(k) for k in range(len(val_acc_lst))], val_acc_lst)
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.title("accuracy on the validation data")
    plt.savefig("#2(c) item_response.png")

    test_acc = evaluate(test_data, theta=theta, beta=beta)
    print("test accuracy:", test_acc)
    #####################################################################
    pass

    # TODO:                                                             #
    # Implement part (d)
    plt.figure(figsize=(20, 18))

    p_i1 = [sigmoid(theta[i] - beta[1]) for i in range(len(theta))]
    p_i2 = [sigmoid(theta[i] - beta[2]) for i in range(len(theta))]
    p_i3 = [sigmoid(theta[i] - beta[3]) for i in range(len(theta))]
    p_i4 = [sigmoid(theta[i] - beta[4]) for i in range(len(theta))]
    p_i5 = [sigmoid(theta[i] - beta[5]) for i in range(len(theta))]
    # smooth p_ij
    p_i1 = signal.savgol_filter(p_i1, 21, 3)
    p_i2 = signal.savgol_filter(p_i2, 21, 3)
    p_i3 = signal.savgol_filter(p_i3, 21, 3)
    p_i4 = signal.savgol_filter(p_i4, 21, 3)
    p_i5 = signal.savgol_filter(p_i5, 21, 3)
    plt.plot([k for k in range(len(p_i1))], p_i1, label='p_i1')
    plt.plot([k for k in range(len(p_i2))], p_i2, label='p_i2')
    plt.plot([k for k in range(len(p_i3))], p_i3, label='p_i3')
    plt.plot([k for k in range(len(p_i4))], p_i4, label='p_i4')
    plt.plot([k for k in range(len(p_i5))], p_i5, label='p_i5')
    plt.xlabel("theta")
    plt.ylabel("P_ij")
    plt.title("P_ij-theta curves")
    plt.legend()
    plt.savefig("#2(d) item_response.png")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
