from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    return acc
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    # user-based collaborative filtering
    k_list = [1, 6, 11, 16, 21, 26]
    user_based_acc = []
    for k in k_list:
        user_based_acc.append(knn_impute_by_user(sparse_matrix, val_data, k))
    plt.figure(figsize=(10, 8))
    plt.plot([str(k) for k in k_list], user_based_acc)
    plt.xlabel("K")
    plt.ylabel("acc")
    plt.title("accuracy on the validation data by user-based collaborative filtering")
    plt.savefig("#1 knn_plot_user_based.png")

    test_acc = knn_impute_by_user(sparse_matrix, test_data, 11)
    print("accuracy on the validation data by user-based collaborative filtering:", test_acc)

    # item-based collaborative filtering
    k_list = [1, 6, 11, 16, 21, 26]
    item_based_acc = []
    for k in k_list:
        item_based_acc.append(knn_impute_by_item(sparse_matrix, val_data, k))
    plt.figure(figsize=(10, 8))
    plt.plot([str(k) for k in k_list], item_based_acc)
    plt.xlabel("K")
    plt.ylabel("acc")
    plt.title("accuracy on the validation data by item-based collaborative filtering")
    plt.savefig("#1 knn_plot_item_based.png")

    test_acc = knn_impute_by_item(sparse_matrix, test_data, 21)
    print("accuracy on the validation data by item-based collaborative filtering:", test_acc)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
