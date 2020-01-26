from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.engine.saving import load_model
import tensorflow
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from pretty_plot import plot_confusion_matrix_from_data

model_name = "mlp_model.h5"


def execute(X_train, Y_train, X_test, Y_test):

    Y_train = np_utils.to_categorical(Y_train)
    model = use_neural_network(X_train, Y_train)
    score = prediction_evaluation(model, X_test, Y_test)

    return model, score


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
    input_dim = X_train.shape[1]
    classes = Y_train.shape[1]

    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    print("Training...")
    model.fit(X_train, Y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=2)

    model.save(model_name)

    return model

def prediction_evaluation(model, X_test, Y_test):

    print("Multilayer perceptron\n")
    #confusion matrix

    predictions = np.argmax(model.predict(X_test),axis=1)
    cnf_matrix = confusion_matrix(Y_test, predictions)

    plot_confusion_matrix_from_data(y_test=Y_test, predictions=predictions, columns=[0,1,2,3,4,5,6,7,8,9])

    print(cnf_matrix)

    #accuracy

    eval = model.evaluate(X_test, np_utils.to_categorical(Y_test), verbose=0, )
    print(f"Accuracy: {eval[1]}")
    return eval[1]
def load_created_model(name):
    return load_model(name)