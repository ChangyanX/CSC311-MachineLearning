'''
Question 2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data


import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))

    # Compute means
    x = train_data
    t = train_labels

    # 1. categorize all pictures into 10 classes
    # 2. find mean estimate for each class (1,64)
    # 3. combine all mean estimate into a matrix => means

    for k in range(10):
        pics_k = data.get_digits_by_label(x, t, k)
        tot_num = len(pics_k)
        tot_pixels = np.zeros((1, 64))
        for i in range(len(pics_k)):
            tot_pixels += pics_k[i]
        means[k] = tot_pixels / tot_num

    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    x = train_data
    t = train_labels
    means = compute_mean_mles(train_data, train_labels) # (10,64)

    # 1. categorize all pictures into 10 classes
    # 2. find sigma estimate for each class (1,64,64)
    # 3. combine all sigma estimate into a matrix => covariances

    for k in range(10):
        mean_k = means[k] # (1, 64)
        tot_pixels_cov = np.zeros((64, 64))
        pics_k = data.get_digits_by_label(x, t, k)
        tot_num = len(pics_k)
        for i in range(len(pics_k)):
            xk_minus_meank = pics_k[i].reshape((64,1)) - mean_k.reshape((64,1))
            tot_pixels_cov += np.matmul(xk_minus_meank, xk_minus_meank.T)
        covariances[k] = 0.01*np.identity(64) + tot_pixels_cov / tot_num

    return covariances


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    x = digits
    d = len(x[0]) # dimension of the data
    result = np.zeros((len(x), 10))
    for k in range(10):
        cov_k = covariances[k]
        mu_k = means[k]
        a = np.zeros((len(x), 1))

        for i in range(len(x)):
            x_minus_mu = x[i] - mu_k
            cov_k_inv = np.linalg.inv(cov_k)
            a[i] = - 1/2 * np.matmul(np.matmul(x_minus_mu.T, cov_k_inv),
                                    x_minus_mu)
            assert a[i].shape == (1,)

        m = np.max(a, axis=None, keepdims=True)
        for i in range(len(x)):
            result[i][k] = np.log((2 * np.pi)**(-d/2)) \
                           + np.log(np.linalg.det(covariances[k])**(-1/2)) \
                           + a[i]
    return result

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    x = digits
    n = len(x)      # size of the dataset
    d = len(x[0])   # dimension of the data
    log_gen_likelihood = generative_likelihood(digits, means, covariances)
    result = np.zeros((n, 10))


    subresult = np.zeros((n, 10))
    for k in range(10):
        cov_k = covariances[k]
        mu_k = means[k]
        a = np.zeros((n, 1))  # shape: (n,1))
        for i in range(n):
            x_minus_mu = x[i] - mu_k
            cov_k_inv = np.linalg.inv(cov_k)
            a[i] = - 1 / 2 * (x_minus_mu.T @ cov_k_inv @ x_minus_mu)
            assert a[i].shape == (1,)
        subresult_T = subresult.T
        subresult_T[k] = a.T
    result = subresult*0.01


    for i in range(n):
        m = np.max(subresult[i], axis=None, keepdims=True)
        for k in range(k):
            result[i][k] -= (np.log(np.sum(np.exp(subresult[i] - m))) + m)*0.01

    return result

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return

    tot = 0
    for i in range(len(digits)):
        k = int(labels[i])
        tot += cond_likelihood[i][k]

    return tot / len(digits)

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    result = np.argmax(cond_likelihood, axis=1).T
    return result

def main():
    train_data, train_labels, test_data, test_labels = \
        data.load_all_data_from_zip('hw4digits.zip', 'data')


    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    # 2(a)
    train_avg_cond = \
        avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_avg_cond = \
        avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print("The avg_conditional_likelihood for TRAIN: \n "
          "{}".format(train_avg_cond))
    print("The avg_conditional_likelihood for TEST: \n "
          "{}".format(test_avg_cond))



    # 2(b) find predictions of training data and test data,
    # and find the accurarcy
    train_predictions = classify_data(train_data, means, covariances)
    test_predictions = classify_data(test_data, means, covariances)
    num_correct_train = np.count_nonzero(train_predictions - train_labels == 0)
    num_correct_test = np.count_nonzero(test_predictions - test_labels == 0)

    print("train accuracy: ", num_correct_train / len(train_data))
    print("test accuracy: ", num_correct_test / len(test_data))



    # 2(c) Compute leading eigenvectors for each class cov matrix
    # plot them side by side as 8 by 8 images
    for k in range(10):
        w, v = np.linalg.eig(covariances[k])
        max_w_index = np.where(w == max(w))
        leading_v = v[:, max_w_index]
        # print("The leading vector for k={} is: \n {}".format(k, leading_v))


        plt.figure()
        im = plt.imshow(leading_v.reshape(8,8), cmap='gray')
        plt.savefig("{}.png".format(k))
        plt.show()


if __name__ == '__main__':
    main()
