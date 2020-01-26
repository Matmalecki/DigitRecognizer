from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.engine.saving import load_model
import tensorflow
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from pretty_plot import plot_confusion_matrix_from_data

model_name = "conv_model.h5"

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1, len(preds) + 1)), "Label": preds}).to_csv(fname, index=False, header=True)

def for_submission(model, X_test):
    preds = model.predict_classes(X_test, verbose=0)

    write_preds(preds, "my_submission_next.csv")

def execute(X_train, Y_train, X_test, Y_test):

    X_train = X_train.reshape(-1,28,28,1)
    X_test = X_test.reshape(-1,28,28,1)


    Y_train = np_utils.to_categorical(Y_train)
    model = use_neural_network(X_train, Y_train)
    score = prediction_evaluation(model, X_test, Y_test)

    return model, score

def execute_for_submission(values, labels):
    values = values.reshape(-1,28,28,1)
    labels = np_utils.to_categorical(labels)

    model = use_neural_network(values, labels)
    test = pd.read_csv('input/test.csv')
    submission_data = test.values.astype('float32')
    submission_data = normalize_data(submission_data)
    submission_data = submission_data.reshape(-1,28,28,1)
    for_submission(model, submission_data)

def use_neural_network(X_train, Y_train):
    #model = create_model(X_train, Y_train)
    model = load_created_model(model_name)
    return model


def normalize_data(data):
    scale = np.max(data)
    data /= scale
    mean = np.std(data)
    data -= mean
    return data

def create_model(X_train, Y_train):

    model = Sequential()
    model.add(Conv2D(32,kernel_size=5,activation='relu',input_shape=(28,28,1)))
    model.add(MaxPool2D())
    model.add(Conv2D(64,kernel_size=5,activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(2**7, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


    print("Training...")
    model.fit(X_train, Y_train, epochs=20, batch_size=64, validation_split=0.1, verbose=2)

    model.save(model_name)

    return model

def prediction_evaluation(model, X_test, Y_test):

    print("Convolutional neural network\n")
    #confusion matrix

    predictions = np.argmax(model.predict(X_test),axis=1)
    cnf_matrix = confusion_matrix(Y_test, predictions)
    #plot_confusion_matrix_from_data(y_test=Y_test, predictions=predictions, columns=[0,1,2,3,4,5,6,7,8,9])
    print(cnf_matrix)

    #accuracy

    eval = model.evaluate(X_test, np_utils.to_categorical(Y_test), verbose=0, )
    print(f"Accuracy: {eval[1]}")
    return eval[1]

def load_created_model(name):
    return load_model(name)