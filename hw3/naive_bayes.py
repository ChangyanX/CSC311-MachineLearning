import numpy as np
import os
import gzip
import struct
import array
import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve


def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images[:1000], test_labels[:1000]


def load_mnist():
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = (partial_flatten(train_images) / 255.0 > .5).astype(float)
    test_images = (partial_flatten(test_images) / 255.0 > .5).astype(float)
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels


def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28, 28),
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[1]] = cur_image
        cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    return cax


def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)


def train_mle_estimator(train_images, train_labels):
    """ Inputs: train_images, train_labels
        Returns the MLE estimators theta_mle and pi_mle"""

    # YOU NEED TO WRITE THIS PART
    x = train_images    # N x D
    t = train_labels    # N x 10

    N_data = len(x)     # size of the data
    D_data = len(x[0])  # dimension of the data


    # find pi_mle
    sum_all_label_pixels = np.sum(t)
    sum_t0_to_t9 = np.zeros((10, 1))
    for i in range(10):
        sum_t0_to_t9[i] = np.sum(t.T[i])

    pi_mle = sum_t0_to_t9 / N_data


    # find theta_mle
    xT = x.T
    tT = t.T
    theta_mle = np.zeros((784, 10))
    for j in range(D_data):
        for c in range(10):
            denominator = np.sum(t.T[c])
            theta_mle[j][c] = tT[c].T.dot(xT[j]) / denominator


    assert pi_mle.shape == (10, 1)
    assert theta_mle.shape == (784, 10)

    return theta_mle, pi_mle


def train_map_estimator(train_images, train_labels):
    """ Inputs: train_images, train_labels
        Returns the MAP estimators theta_map and pi_map"""
    
    # YOU NEED TO WRITE THIS PART
    a, b = 3, 3       # beta(3, 3)
    x = train_images  # N x D
    t = train_labels  # N x 10

    N_data = len(x)  # size of the data
    D_data = len(x[0])  # dimension of the data


    theta_nominator = x.T.dot(t) + 2
    theta_denominator = x.T.dot(t) + (np.ones((N_data, D_data))).T.dot(t) + 4
    theta_map = theta_nominator / theta_denominator
    assert theta_map.shape == (784, 10)

    pi_map = train_mle_estimator(train_images, train_labels)[1]

    return theta_map, pi_map


def log_likelihood(images, theta, pi):
    """ Inputs: images, theta, pi
        Returns the matrix 'log_like' of loglikehoods over the input images where
    log_like[i,c] = log p (c |x^(i), theta, pi) using the estimators theta and pi.
    log_like is a matrix of num of images x num of classes
    Note that log likelihood is not only for c^(i), it is for all possible c's."""

    # YOU NEED TO WRITE THIS PART
    x = images
    N_data = len(x)

    log_like = np.zeros((N_data, 10))

    for i in range(N_data):
        image = x[i]

        one_minues_xj = np.ones((1, 784)) - image
        one_minues_thetajc = np.ones((784, 10)) - theta

        # prevent from log(0) = -inf
        for i in range(784):
            for j in range(10):
                if one_minues_thetajc[i][j] == 0:
                    one_minues_thetajc[i][j] = 1

        log_one_minues_thetajc = np.log(one_minues_thetajc)

        x_dot_log_theta = image.dot(np.log(theta))
        oneMinusx_dot_log_oneMinustheta = \
            one_minues_xj.dot(log_one_minues_thetajc)


        marginal_prob = np.sum(x_dot_log_theta) + \
                        np.sum(oneMinusx_dot_log_oneMinustheta)

        log_like[i] = np.log(pi).T + x_dot_log_theta \
                   + oneMinusx_dot_log_oneMinustheta - marginal_prob
        #print(oneMinusx_dot_log_oneMinustheta)
    assert log_like.shape == (N_data, 10)

    return log_like


def predict(log_like):
    """ Inputs: matrix of log likelihoods
    Returns the predictions based on log likelihood values"""

    # YOU NEED TO WRITE THIS PART
    predictions = np.zeros((len(log_like), 10))
    for i in range(len(log_like)):
        idx = np.argmax(log_like[i])
        predictions[i][idx] = 1

    assert predictions.shape == (len(log_like), 10)
    return predictions


def accuracy(log_like, labels):
    """ Inputs: matrix of log likelihoods and 1-of-K labels
    Returns the accuracy based on predictions from log likelihood values"""

    # YOU NEED TO WRITE THIS PART
    predictions = predict(log_like)

    diff = predictions - labels
    all_10_zeros = np.zeros((1,10))

    correct = 0
    for i in range(len(diff)):
        if np.array_equal(diff[i], all_10_zeros):
            correct += 1
            print(correct)

    total = len(log_like)
    accuracy = correct / total
    return accuracy


def image_sampler(theta, pi, num_images):
    """ Inputs: parameters theta and pi, and number of images to sample
    Returns the sampled images"""

    # YOU NEED TO WRITE THIS PART

    return sampled_images


def main():
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()

    # Fit MLE and MAP estimators
    theta_mle, pi_mle = train_mle_estimator(train_images, train_labels)
    theta_map, pi_map = train_map_estimator(train_images, train_labels)

    # Find the log likelihood of each data point
    loglike_train_mle = log_likelihood(train_images, theta_mle, pi_mle)
    loglike_train_map = log_likelihood(train_images, theta_map, pi_map)

    avg_loglike_mle = np.sum(loglike_train_mle * train_labels) / N_data
    avg_loglike_map = np.sum(loglike_train_map * train_labels) / N_data

    print("Average log-likelihood for MLE is ", avg_loglike_mle)
    print("Average log-likelihood for MAP is ", avg_loglike_map)

    train_accuracy_map = accuracy(loglike_train_map, train_labels)
    loglike_test_map = log_likelihood(test_images, theta_map, pi_map)
    test_accuracy_map = accuracy(loglike_test_map, test_labels)

    print("Training accuracy for MAP is ", train_accuracy_map)
    print("Test accuracy for MAP is ", test_accuracy_map)

    # Plot MLE and MAP estimators
    save_images(theta_mle.T, 'mle.png')
    save_images(theta_map.T, 'map.png')

    # Sample 10 images
    sampled_images = image_sampler(theta_map, pi_map, 10)
    save_images(sampled_images, 'sampled_images.png')


if __name__ == '__main__':
    main()
