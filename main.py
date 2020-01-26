import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import multi_layer_perceptron as mlp
import decision_tree as dec
import nearest_neighbors as neigh
import convolutional_neural_network as conv
import random_forest as rnf
import matplotlib.pyplot as plt

def for_submission(values, labels):
    conv.execute_for_submission(values, labels)

def chart_digits(data, set="treningowym"):
    x = [0] * 10
    for val in data:
        x[val]+=1

    labels = [0,1,2,3,4,5,6,7,8,9]
    plt.bar(labels, x, align='center', alpha=0.5)
    plt.xticks(labels, labels)
    plt.ylabel(f'Ilość danych w zbiorze {set}')
    plt.title('Ilość każdej z cyfr')

    plt.savefig(f'count_of_digits+{set}.png')

def chart_accuracy_of_algs(data):
    D = data
    plt.bar(range(len(D)), list(D.values()), align='center')
    plt.xticks(range(len(D)), list(D.keys()))

    plt.savefig("accuracy_result.png")
    plt.show()

def normalize_data(data):
    scale = np.max(data)
    data /= scale
    mean = np.std(data)
    data -= mean
    return data

def main():
    train = pd.read_csv('input/train.csv')
    labels = train["label"].values.astype('int32')
    values = (train.drop("label", axis=1).values).astype('float32')

    #for_submission(normalize_data(values), labels)
    X_train, X_test, Y_train, Y_test = train_test_split(values, labels, test_size=0.1)

    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)

    print (f"Rozmiar testowych danych {len(Y_test)}");
    acc = 0
    accuracies = dict()
    # multi_layer_perceptron

    model, acc = mlp.execute(X_train, Y_train, X_test, Y_test)

    accuracies["MLP"] = acc * 100

    #convolutional neural network
    model_conv, acc = conv.execute(X_train, Y_train, X_test, Y_test)

    accuracies["CNN"] = acc * 100
    #decision tree
    acc = dec.execute(X_train, Y_train, X_test, Y_test)

    accuracies["D-Tree"] = acc * 100
    #nearest neighbors
    acc = neigh.execute(X_train, Y_train, X_test, Y_test)

    accuracies["5-NN"] = acc * 100
    #random forest
    acc = rnf.execute(X_train, Y_train, X_test, Y_test)

    accuracies["R-Forest"] = acc * 100

    chart_accuracy_of_algs(accuracies)
    chart_digits(Y_train)
    chart_digits(Y_test, 'testowym')

if __name__ == "__main__":
    main()

