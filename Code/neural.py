import pandas as pd
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
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
    Ytrain_hyperp, Ytest_hyperp = training_data['hyperp'], test_data['hyperp']
    Ytrain_bias, Ytest_bias = training_data['bias'], test_data['bias']

    print("length of train set:", len(Xtrain))
    print("length of test set:", len(Xtest))

    vectorizer = TfidfVectorizer()
    encoder = LabelEncoder()
    vectorizer.fit(Xtrain)

    Xtrain = vectorizer.transform(Xtrain)
    Xtest  = vectorizer.transform(Xtest)
    Ytrain_hyperp = encoder.fit_transform(Ytrain_hyperp)
    Ytest_hyperp = encoder.fit_transform(Ytest_hyperp)
    #Ytrain = to_categorical(Ytrain)
    #Ytest = to_categorical(Ytest)

    input_dim = Xtrain.shape[1]

    # Hyperp
    model = Sequential()
    model.add(layers.Dense(10, input_dim = input_dim, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(Xtrain, Ytrain_hyperp,epochs = 1 ,verbose=True, validation_data=(Xtest, Ytest_hyperp), batch_size=32)

    # Bias
    classifier = LinearSVC(C=1)
    classifier.fit(Xtrain, Ytrain_bias)
    Yguess = classifier.predict(Xtest)
    print("Accuracy of bias {}".format(accuracy_score(Yguess, Ytest_bias)))

    yhat_probs = model.predict(Xtest, verbose=0)
    yhat_classes = model.predict_classes(Xtest, verbose=0)

    bias_prediction_list = Yguess
    hyperp_prediction_list = []
    id_list = test_data['id'].tolist()

    for i in yhat_classes:
        if i == 0:
            i = "false"
            hyperp_prediction_list.append(i)
        else:
            i = "true"
            hyperp_prediction_list.append(i)

    output = list(zip(id_list, hyperp_prediction_list, bias_prediction_list))
    for i in output:
        print("{} {} {}".format(i[0], i[1], i[2]))

    """
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(Ytest_hyperp, yhat_classes)
    print("Accuracy of hyperp {}".format(accuracy))
    """
    """
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
