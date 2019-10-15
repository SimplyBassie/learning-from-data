import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


train = pd.read_csv('../Data/hyperp-training-grouped.csv.xz',
                    compression='xz',
                    sep='\t',
                    encoding='utf-8',
                    index_col=0).dropna()

train.sample(3)
pipeline = Pipeline([('vec', CountVectorizer()),
                    ('clf', MultinomialNB())])
model = pipeline.fit(train.text, train.hyperp)
y_pred = model.predict(train.text)
print(accuracy_score(y_pred, train.hyperp))
