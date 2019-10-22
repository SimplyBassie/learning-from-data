from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import sys
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import ShuffleSplit
#from keras.models import Sequential
#from keras import layers



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

def identity(x):
    return x

def preprocess(tweet):
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    pptweetlist = []
    tweet = stemmer.stem(tweet)
    tweet = lemmatizer.lemmatize(tweet)
    return tweet

def tokenize(tweet):
    tknzr = TweetTokenizer()
    stop_words = stopwords.words('english')
    wordlist = tknzr.tokenize(tweet)
    wordlistwithoutstopwords = []
    for word in wordlist:
        if word not in stop_words:
            wordlistwithoutstopwords.append(word)
    toktweet = " ".join(wordlistwithoutstopwords)

    return wordlistwithoutstopwords


def print_n_most_informative_features(coefs, features, n):
    # Prints the n most informative features
    most_informative_feature_list = [(coefs[0][nr],feature) for nr, feature in enumerate(features)]
    sorted_most_informative_feature_list = sorted(most_informative_feature_list, key=lambda tup: abs(tup[0]), reverse=True)
    print("\nMOST INFORMATIVE FEATURES\n#\tvalue\tfeature")
    for nr, most_informative_feature in enumerate(sorted_most_informative_feature_list[:n]):
        print(str(nr+1) + ".","\t%.3f\t%s" % (most_informative_feature[0], most_informative_feature[1]))


def main():

    trainfile = sys.argv[1]
    testfile = sys.argv[2]

    training_data, test_data = read_data(trainfile, testfile)
    Xtrain, Xtest = training_data['text'], test_data['text']
    Ytrain, Ytest = training_data['bias'], test_data['bias']

    print("length of train set:", len(Xtrain))
    print("length of test set:", len(Xtest))



    #vec = TfidfVectorizer(tokenizer = tokenize,
    #                      preprocessor = preprocess,
    #                      ngram_range = (1,5))
    vec = TfidfVectorizer()

    clf1 = DecisionTreeClassifier(max_depth=20)
    clf2 = KNeighborsClassifier(n_neighbors=9)
    clf3 = LinearSVC(C=1)
    #ens = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='hard', weights=[1, 1, 1])
    clf4 = MultinomialNB()
    classifier = Pipeline( [('vec', vec), ('cls', clf3)] )

    classifier.fit(Xtrain, Ytrain)

    coefs = classifier.named_steps['cls'].coef_
    features = classifier.named_steps['vec'].get_feature_names()
    print_n_most_informative_features(coefs, features, 10)

    Yguess = classifier.predict(Xtest)

    print(classification_report(Ytest, Yguess))
    print(accuracy_score(Yguess, Ytest))

if __name__ == '__main__':
    main()
