import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# from sklearn.model_selection import KFold
# from sklearn.utils import shuffle

# ==============================================================
# Q4(a) Import data
# ==============================================================
file_path = "/Users/ChangyanXu/Desktop/CSC311/HW1/data/"

data_train = {'X': np.genfromtxt(file_path+'data_train_X.csv', delimiter=','),
              't': np.genfromtxt(file_path+'data_train_y.csv', delimiter=',')}

data_test = {'X': np.genfromtxt(file_path+'data_test_X.csv', delimiter=','),
             't': np.genfromtxt(file_path+'data_test_y.csv', delimiter=',')}

# 'X': (n_samples=380, n_features=400)
# 't': (n_samples=380)

# ==============================================================
# Q4(b) Finish 6 functions for cross validation
# ==============================================================
def shuffle_data(data):
    """
    Returns randomly permuted version (of copy) of data along the samples

    :param data: A dictionary in the form of {'X': X, 't': t}, where X is an
    ndarray of (n_sample, n_feature), and t is a vector of (n_sample)
    :return: A dictionary in the form of {'X': X, 't': t}, just as data
    """
    X = data['X'].copy()
    t = data['t'].copy()

    idx = np.arange(len(X))
    np.random.shuffle(idx)

    X = X[idx]
    t = t[idx]
    return {'X': X, 't': t}


def split_data(data, num_folds: int, fold: int):
    """

    :param data: A dictionary in the form of {'X': X, 't': t}, where X is an
    ndarray of (n_sample, n_feature) and t is a vector of (n_sample)
    :param num_folds: (int) number of partitions
    :param fold: the selected partition, <fold> should be in {1, ..., k}
    :return: a tuple of the selected partition <fold> as <data_fold> and the
    remaining data as <data_rest>

    Note: also can use np.array_split
    """
    X = data['X'].copy()
    t = data['t'].copy()

    if fold > num_folds or fold < 1:
        raise ValueError("<fold> should be in {1, ..., k}")


    width = int(len(X) / num_folds)
    for i, start in enumerate(range(0, len(X), width)):
        end = start + width
        if fold == i+1:
            data_fold = {'X': X[start:end],
                         't': t[start:end]}
            data_rest = {'X': np.concatenate((X[:start], X[end:])),
                         't': np.concatenate((t[:start], t[end:]))}

    return data_fold, data_rest


def train_model(data, lambd):
    """
    Return the coefficients, w, of ridge regression with penalty level <lambd>

    Note: ignore bias parameter; (see w_ridge in W2 slides pg.33)
    """
    X = data['X'].copy()
    t = data['t'].copy()

    N = len(X)

    within_inverse = np.dot(X.transpose(), X) + lambd*N*np.identity(len(X[0]))
    inverse_part = np.linalg.inv(within_inverse)

    return np.dot(np.dot(inverse_part, X.transpose()), t)


def predict(data, model):
    """
    Return the prediction y by predicting with linear regression y = Xw.
    """
    return np.dot(data['X'], model)


def loss(data, model):
    """
    Return the average squared error loss based on <model>.
    """
    X = data['X'].copy()
    t = data['t'].copy() # shape: (n_features, )
    y = np.dot(X, model) # predictions; shape: (n_features, )

    error_matrix = np.matrix(y - t) # shape: (1, n_features)
    result = 1/(2*len(X)) * np.dot(error_matrix, error_matrix.transpose())[0][0]
    return float(result)


def cross_validation(data, num_folds, lambd_seq):
    """Return the cross validation error across all lambds

    lambd_seq: an evenly spaced 50 numbers over the interval (0.00005,
    0.005), including both ends

    lambd_seq = np.linspace(0.00005, 0.005, num=50, endpoint=True)
    """
    assert len(lambd_seq) == 50

    cv_errors = []
    data_shf = shuffle_data(data)
    for i, lambd in enumerate(lambd_seq):
        cv_loss_lmd = 0
        for j, fold in enumerate(range(1, num_folds)):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_errors.append(cv_loss_lmd / num_folds)

    return np.array(cv_errors)

# ==============================================================
# Q4(c) Report the training and test errors corresponding to
# each lambda in <lambd_seq>
# ==============================================================
# generate a plot: training and test errors vs. <lambd_seq>
lambd_seq = np.linspace(0.00005, 0.005, num=50, endpoint=True)
training_errors = []
test_errors = []
for lambd in lambd_seq:
    w_model = train_model(data_train, lambd)
    training_errors.append(loss(data_train, w_model))
    test_errors.append(loss(data_test, w_model))

idx_min = test_errors.index(min(test_errors))
opt_lambd = lambd_seq[idx_min]
print("=== Q4(c) ===")
print("index of min test_error: ", idx_min)
print("min test_error: ", test_errors[idx_min])
print("optimal lambd: ", opt_lambd)

plt.figure()
plt.plot(lambd_seq, training_errors, '*-c', label='training error')
plt.plot(lambd_seq, test_errors, 'x-r', label='test error')
plt.axvline(x=opt_lambd, color='k', linestyle='-',
            label='optimum lambd ≈ {}'.format(round(opt_lambd, 5)))
plt.title('Q1(c) Training and Test Errors vs. lambd')
plt.xlabel('lambd (np.linspace(0.00005, 0.005, num=50))')
plt.ylabel('Training and Test Errors')
plt.legend()
plt.grid()
plt.savefig("Q4(c) training and test errors vs. <lambd_seq>.png")



# ==============================================================
# Q4(d)
# 1. Plot a graph with
#       - training error
#       - test error
#       - 5-fold cross validation errors
#       - 10-fold cross validation errors
#    vs. <lambd_seq>
# 2. Find lambd proposed by cross validation procedure
# 3. Comment on the shapes of the error values
# ==============================================================
cv_error_5_fold = cross_validation(data_train, 5, lambd_seq).tolist()
cv_error_10_fold = cross_validation(data_train, 10, lambd_seq).tolist()


# smallest cv error in 5-fold cv
idx_5 = cv_error_5_fold.index(min(cv_error_5_fold))
opt_lambd_5 = lambd_seq[idx_5]
print("=== Q4(d) 5-fold ===")
print("index of min cv_error: ", idx_5)
print("min 5_fold cv_error: ", cv_error_5_fold[idx_5])
print("optimal lambd: ", opt_lambd_5)


# smallest cv error in 10-fold cv
idx_10 = cv_error_10_fold.index(min(cv_error_10_fold))
opt_lambd_10 = lambd_seq[idx_10]
print("=== Q4(d) 10-fold ===")
print("index of min cv_error: ", idx_10)
print("min 10_fold cv_error: ", cv_error_10_fold[idx_10])
print("optimal lambd: ", opt_lambd_10)


plt.figure()
plt.plot(lambd_seq, training_errors, '*-c', label='training error')
plt.plot(lambd_seq, test_errors, 'x-r', label='test error')
plt.plot(lambd_seq, cv_error_5_fold, '*-g', label='5-Fold CV_Errors')
plt.plot(lambd_seq, cv_error_10_fold, '*-b', label='10-Fold CV_Errors')

plt.axvline(x=opt_lambd_10, color='k', linestyle='-',
            label='optimal lambd with smallest 5-fold cv_error ≈ {}'.format(
                round(opt_lambd_5, 5)))
plt.title('Q4(d) training errors, test errors, 5-fold and 10-fold '
          'cv_errors vs. lambd')
plt.xlabel('lambd (np.linspace(0.00005, 0.005, num=50))')
plt.ylabel('errors')
plt.legend()
plt.grid()
plt.savefig("Q4(d) training errors, test errors, 5-fold cv_errors, 10-fold "
            "cv_errors vs. <lambd_seq>.png")
