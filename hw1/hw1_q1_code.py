# scipy numpy autograd matplotlib jupyter sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
# from sklearn.decomposition import PCA
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler


clean_fake_path = "/Users/ChangyanXu/Desktop/CSC311/HW1/data/clean_fake.txt"
clean_real_path = "/Users/ChangyanXu/Desktop/CSC311/HW1/data/clean_real.txt"


# ======= Q1 (a) =======
def load_data(real_path: str, fake_path: str):
    """
    Load .txt files of real data and fake data. Preprocesses with
    CountVectorizer and split the entire data set randomly into 70% training,
    15% validation, and 15% test examples.
    """

    # load txt file, forms a dataset
    dataset = []
    real, fake = real_path, fake_path # real with label 1, fake with label 0

    file = open(real, 'r')
    lines = file.readlines()
    for i in lines:
        j = i.replace('\n', '')
        dataset.append([j, 1]) # real -- 1
    file.close()

    file = open(fake, 'r')
    lines = file.readlines()
    for i in lines:
        j = i.replace('\n', '')
        dataset.append([j, 0])  # fake -- 0
    file.close()

    dataset = np.array(dataset)
    X, y = dataset[:, 0], dataset[:, 1]


    # preprocesses with a CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)


    # split dataset randomly - 70% training, 15% validation, 15% test examples
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=0.7, random_state=42, shuffle=True) # shuffle the data before splitting
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=42, shuffle=True) # shuffle the data before splitting


    return X_train, X_validation, X_test, y_train, y_validation, y_test


# ======= Q1 (b,c) =======
def select_knn_model(metric, X_train, X_validation, X_test, y_train,
                     y_validation, y_test):

    # try k=1 to k=20, and record the training accuracy and validation
    # accurarcy scores
    k_range = range(1, 21)
    training_accuracy = []
    validation_accuracy = []

    # train the model with training dataset
    for k in k_range:
        if metric == 'cosine':
            knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        else:
            knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # make prediction on y_train, and find accuracy_score
        y_train_predict = knn.predict(X_train)
        assert len(y_train_predict) == len(y_train)
        training_accuracy.append(
            # knn.score(X_train, y_train)
            metrics.accuracy_score(y_train, y_train_predict)
        )

        # make prediction on y_validation, and find accuracy_score
        y_validation_predict = knn.predict(X_validation)
        assert len(y_validation_predict) == len(y_validation)
        validation_accuracy.append(
            # knn.score(X_validation, y_validation)
            metrics.accuracy_score(y_validation, y_validation_predict)
        )


    # generate a plot: training and validation accuracy vs. k
    plt.figure()
    plt.plot(k_range, training_accuracy, '*-c', label='training accuracy')
    plt.plot(k_range, validation_accuracy, 'x-r', label='validation accuracy')
    plt.xlabel('k')
    plt.ylabel('training and validation accuracy')
    plt.legend()
    plt.grid()
    if metric == 'cosine':
        plt.title('Q1(c) Training and Validation Accuracy vs. k-value ('
                  'metric=\'cosine\')')
        plt.savefig("Q1(c) Training and Validation Accuracy vs. k-value ("
                    "metric='cosine').png")
    else:
        plt.title('Q1(b) Training and Validation Accuracy vs. k-value')
        plt.savefig("Q1(b) Training and Validation Accuracy vs. k-value.png")


    # choose the k with best validation accuracy
    # and report the accuracy on test data
    k_best = validation_accuracy.index(max(validation_accuracy)) + 1
    if metric == 'cosine':
        knn_best = KNeighborsClassifier(n_neighbors=k_best, metric='cosine')
    else:
        knn_best = KNeighborsClassifier(n_neighbors=k_best)
    knn_best.fit(X_train, y_train)

    y_test_predict = knn_best.predict(X_test)
    assert len(y_test_predict) == len(y_test)
    test_score = metrics.accuracy_score(y_test, y_test_predict)

    return k_best, test_score


if __name__ == '__main__':
    # Q1(a)
    X_train, X_validation, X_test, y_train, y_validation, y_test = \
        load_data(clean_real_path, clean_fake_path)



    # Q1(b)
    k_best, test_score = select_knn_model('default', X_train, X_validation,
                                          X_test, y_train, y_validation, y_test)
    print("(b) k_best: {}, test_score: {}".format(k_best, test_score))
        # result:
        # k_best: 16, test_score: 0.7



    # Q1(c)
    k_best_c, test_score_c = select_knn_model('cosine', X_train, X_validation,
                                              X_test, y_train, y_validation,
                                              y_test)
    print("(c) k_best: {}, test_score: {}".format(k_best_c, test_score_c))


