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

def read_data():
    data = pd.read_csv('../Data/hyperp-training-grouped.csv.xz',
                        compression='xz',
                        sep='\t',
                        encoding='utf-8',
                        index_col=0).dropna()

    data1 = data[(data.index < np.percentile(data.index, 10))] #smaller dataset
    data2 = data[(data.index > np.percentile(data.index, 90))] #smaller dataset
    data = pd.concat([data1, data2])
    return data

def main():
    data = read_data()
    X = np.array(data['text'].tolist())
    Y = np.array(data['hyperp'].tolist())

    kf = ShuffleSplit(n_splits=1, test_size=0.1)
    for train_index, test_index in kf.split(X):
        #print(train_index, test_index)
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]

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

        model.compile(loss = 'mean_squared_error', optimizer='adam', metrics=['accuracy'])
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
