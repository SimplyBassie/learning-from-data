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
    Ytrain, Ytest = training_data['hyperp'], test_data['hyperp']

    print("length of train set:", len(Xtrain))
    print("length of test set:", len(Xtest))

    vectorizer = CountVectorizer()
    encoder = LabelEncoder()
    vectorizer.fit(Xtrain)

    Xtrain = vectorizer.transform(Xtrain)
    Xtest  = vectorizer.transform(Xtest)
    Ytrain = encoder.fit_transform(Ytrain)
    Ytest = encoder.fit_transform(Ytest)
    #Ytrain = to_categorical(Ytrain)
    #Ytest = to_categorical(Ytest)

    input_dim = Xtrain.shape[1]

    model = Sequential()
    model.add(layers.Dense(10, input_dim = input_dim, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    history = model.fit(Xtrain, Ytrain,epochs = 5 ,verbose=True, validation_data=(Xtest, Ytest), batch_size=30)

    """
    yhat_probs = model.predict(Xtest, verbose=0)
    yhat_classes = model.predict_classes(Xtest, verbose=0)

    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(Ytest, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(Ytest, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(Ytest, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Ytest, yhat_classes)
    print('F1 score: %f' % f1)
    """

if __name__ == '__main__':
    main()

# https://realpython.com/python-keras-text-classification/
