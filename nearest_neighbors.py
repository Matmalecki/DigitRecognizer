import numpy as np
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from sklearn.neighbors import KNeighborsClassifier
from pretty_plot import plot_confusion_matrix_from_data

def execute(X_train, Y_train, X_test, Y_test):

    model = create_model(X_train, Y_train)

    return evaluation(model, X_test, Y_test)

def create_model(X_train, Y_train, neigbors = 3):
    model = KNeighborsClassifier(n_neighbors=5)

    model.fit(X_train, Y_train)

    return model


def evaluation(model, X_test, Y_test):

    print("Nearest neighbours\n")

    predictions = model.predict(X_test)
    #cnf_matrix = confusion_matrix(Y_test, predictions)
    plot_confusion_matrix_from_data(y_test=Y_test, predictions=predictions, columns=[0,1,2,3,4,5,6,7,8,9])
   # print(cnf_matrix)
    score = model.score(X_test, Y_test)
    print("Dokladnosc: {}".format(score))

    return score

