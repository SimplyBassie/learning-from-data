import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np


data = pd.read_csv('../Data/hyperp-training-grouped.csv.xz',
                    compression='xz',
                    sep='\t',
                    encoding='utf-8',
                    index_col=0).dropna()

training_data = data[(data.index < np.percentile(data.index, 80))]
testing_data = data[(data.index > np.percentile(data.index, 80))]

Xtrain, Ytrain = training_data['text'], training_data['hyperp']
Xtest, Ytest = testing_data['text'], testing_data['hyperp']

pipeline = Pipeline([('vec', CountVectorizer()), ('clf', MultinomialNB())])
model = pipeline.fit(Xtrain, Ytrain)
prediction = model.predict(Xtest)
print(prediction)
print(accuracy_score(prediction, Ytest))
