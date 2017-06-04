#!/usr/bin/env python

from sklearn import svm 
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

X_train = []
y_train = []

X_test = []
y_test = []

authorlist = ['Rabindranath', 'SaratChandra', 'BankimChandra']

with open('literary/labels.txt', 'r') as labelFile:
    for line in labelFile:
        author, filename = line.split()
        with open('literary/data/' + filename, 'r') as f:
            if "train" in filename:
                X_train.append(f.read())
                y_train.append(authorlist.index(author))
            elif "test" in filename:
                X_test.append(f.read())
                y_test.append(authorlist.index(author))

with open('stopwords_list_ben.txt', 'r') as stopFile:
    stopWords = stopFile.read().split()
    print stopWords

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()
feature_names = vectorizer.get_feature_names()

clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print np.mean((y_test-y_pred)==0)