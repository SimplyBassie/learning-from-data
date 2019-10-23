#!/usr/bin/env python3
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils, generic_utils
np.random.seed(2018)  # for reproducibility and comparability, don't change!
import json
from sklearn.preprocessing import label_binarize
from keras.optimizers import Adamax
import pandas as pd
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from keras import layers
from keras import losses
from keras import optimizers
import sys
import numpy as np
from keras.utils import to_categorical

def read_data(trainfile, testfile):
    training_data = pd.read_csv(trainfile,
                        sep='\t',
                        encoding='utf-8',
                        index_col=0).dropna()
    test_data = pd.read_csv(testfile,
                        sep='\t',
                        encoding='utf-8',
                        index_col=0).dropna()

    return training_data, test_data


def main():
    trainfile = sys.argv[1]
    testfile = sys.argv[2]

    training_data, test_data = read_data(trainfile, testfile)
    Xtrain, Xtest = training_data['text'], test_data['text']
    Ytrain, Ytest = training_data['bias'], test_data['bias']
    #Xdev, Ydev = Xtrain[:20], Ytrain[:20]
    #Xtrain, Ytrain = Xtrain[20:], Ytrain[20:]
    classes = sorted(list(set(Ytrain)))
    # Convert string labels to one-hot vectors
    Ytrain = label_binarize(Ytrain, classes)
    Ytrain = np.array(Ytrain)
    print("length of train set:", len(Xtrain))
    print("length of test set:", len(Xtest))
    vectorizer = CountVectorizer()
    encoder = LabelEncoder()
    vectorizer.fit(Xtrain)

    Xtrain = vectorizer.transform(Xtrain)
    Xtest  = vectorizer.transform(Xtest)
    print(Xtrain)
    print(Xtrain.shape, Ytrain.shape)
    nb_features = Xtrain.shape[1]
    print(nb_features, 'features')
    nb_classes = Ytrain.shape[1]
    print(nb_classes, 'classes')
    # Build the model
    #print(len(Xtrain), len(Ytrain))
    print("Building model...")
    model = Sequential()
    # Single 500-neuron hidden layer with sigmoid activation
    model.add(Dense(input_dim = nb_features, units = 100, activation = 'relu'))
    # Output layer with softmax activation
    model.add(Dense(units = nb_classes, activation = 'sigmoid'))
    # Specify optimizer, loss and validation metric
    model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
    # Train the model
    history = model.fit(Xtrain, Ytrain, epochs = 20, batch_size = 32, validation_data=(Xtest, Ytest), shuffle = True, verbose = 1)

    # Predict labels for test set
    outputs = model.predict(Xtest, batch_size=32)
    pred_classes = np.argmax(outputs, axis=1)

if __name__ == '__main__':
    main()
