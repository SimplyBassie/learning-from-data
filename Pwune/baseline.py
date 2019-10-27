import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sys
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


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

def preprocess(tweet):
    lemmatizer = WordNetLemmatizer()
    pptweetlist = []
    for t in tweet.split():
        pptweetlist.append(lemmatizer.lemmatize(t))
    return " ".join(pptweetlist)

def tokenize(tweet):
    stop_words = stopwords.words('english')
    wordlist = word_tokenize(tweet)
    wordlistwithoutstopwords = []
    for word in wordlist:
        if word not in stop_words:
            wordlistwithoutstopwords.append(word)
    #toktweet = " ".join(wordlistwithoutstopwords)
    return wordlistwithoutstopwords

def main():
    trainfile = sys.argv[1]
    testfile = sys.argv[2]

    training_data, test_data = read_data(trainfile, testfile)
    Xtrain_hyperp, Xtest_hyperp = training_data['text'], test_data['text']
    Ytrain_hyperp, Ytest_hyperp = training_data['hyperp'], test_data['hyperp']
    Xtrain_bias, Xtest_bias = training_data['text'], test_data['text']
    Ytrain_bias, Ytest_bias = training_data['bias'], test_data['bias']

    #vec = CountVectorizer() #voor snel testen
    vec = CountVectorizer()
    clf = MultinomialNB()
    classifier = Pipeline( [('vec', vec), ('cls', clf)] )

    # Hyperp
    classifier.fit(Xtrain_hyperp, Ytrain_hyperp)
    Yguess_hyperp = classifier.predict(Xtest_hyperp)
    print(classification_report(Ytest_hyperp, Yguess_hyperp))

    # Bias
    classifier.fit(Xtrain_bias, Ytrain_bias)
    Yguess_bias = classifier.predict(Xtest_bias)
    x = 0
    for i in Yguess_bias:
        if Yguess_hyperp[x] == False and i in ['right', 'left']:
            Yguess_bias[x] = 'least'
        if Yguess_hyperp[x] == True and i in ['least', 'left-center', 'right-center']:
            Yguess_bias[x] = 'right'
        x += 1
    print(classification_report(Ytest_bias, Yguess_bias))

    bias_prediction_list = Yguess_bias
    hyperp_prediction_list = Yguess_hyperp
    id_list = test_data['id'].tolist()


    joint_label_prediction_list = []
    # Output to file
    output = list(zip(id_list, hyperp_prediction_list, bias_prediction_list))
    with open('output.txt', 'w') as f:
        for i in output:
            print("{} {} {}".format(i[0], str(i[1]).lower(), i[2]), file = f)
            joint_label_prediction_list.append(str(i[1]).lower() + " " + i[2])

    joint_label_gold = list(zip(Ytest_hyperp,Ytest_bias))
    joint_label_gold_string = []
    for i in joint_label_gold:
        joint_label_gold_string.append(str(i[0]).lower() + " " + i[1])
    print(classification_report(joint_label_prediction_list, joint_label_gold_string))

if __name__ == '__main__':
    main()

# https://realpython.com/python-keras-text-classification/
