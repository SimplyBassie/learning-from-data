import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import sys
import numpy as np
import csv

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
    A1 = np.array(data[data.columns[0]].tolist())
    A = np.array(data['id'].tolist())
    B = np.array(data['hyperp'].tolist())
    C = np.array(data['bias'].tolist())
    D = np.array(data['url'].tolist())
    E = np.array(data['labeledby'].tolist())
    F = np.array(data['publisher'].tolist())
    G = np.array(data['date'].tolist())
    H = np.array(data['title'].tolist())
    I = np.array(data['text'].tolist())
    J = np.array(data['raw_text'].tolist())


    kf = ShuffleSplit(n_splits=1, test_size=0.1)
    for train_index, test_index in kf.split(A):
        #print(train_index, test_index)
        A1train, A1test = A1[train_index], A1[test_index]
        Atrain, Atest = A[train_index], A[test_index]
        Btrain, Btest = B[train_index], B[test_index]
        Ctrain, Ctest = C[train_index], C[test_index]
        Dtrain, Dtest = D[train_index], D[test_index]
        Etrain, Etest = E[train_index], E[test_index]
        Ftrain, Ftest = F[train_index], F[test_index]
        Gtrain, Gtest = G[train_index], G[test_index]
        Htrain, Htest = H[train_index], H[test_index]
        Itrain, Itest = I[train_index], I[test_index]
        Jtrain, Jtest = J[train_index], J[test_index]
        with open('training_set_small.csv', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            csv_writer.writerow(['','id', 'hyperp', 'bias', 'url', 'labeledby', 'publisher', 'date', 'title', 'text', 'raw_text'])
            for i in range(len(Atrain)):
                csv_writer.writerow([A1train[i],Atrain[i],Btrain[i],Ctrain[i],Dtrain[i],Etrain[i],Ftrain[i],Gtrain[i],Htrain[i],Itrain[i],Jtrain[i]])
        with open('test_set_small.csv', mode='w') as csv_file2:
            csv_writer2 = csv.writer(csv_file2, delimiter='\t')
            csv_writer2.writerow(['','id', 'hyperp', 'bias', 'url', 'labeledby', 'publisher', 'date', 'title', 'text', 'raw_text'])
            for i in range(len(Atest)):
                csv_writer2.writerow([A1test[i],Atest[i],Btest[i],Ctest[i],Dtest[i],Etest[i],Ftest[i],Gtest[i],Htest[i],Itest[i],Jtest[i]])




if __name__ == '__main__':
    main()

# https://realpython.com/python-keras-text-classification/
