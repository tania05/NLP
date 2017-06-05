#!/usr/bin/env python
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer

X_train = []
y_train = []

X_test = []
y_test = []

authorlist = ['Rabindranath', 'SaratChandra', 'BankimChandra']

with open("../Dataset/literary/literary/labels.txt", 'r') as labelFile:
    for line in labelFile:
        author, filename = line.split()
        with open("../Dataset/literary/literary/data/" + filename, 'r') as f:
            if "train" in filename:
                X_train.append(f.read())
                y_train.append(authorlist.index(author))
            elif "test" in filename:
                X_test.append(f.read())
                y_test.append(authorlist.index(author))

# with open('stopwords_list_ben.txt', 'r') as stopFile:
    # stopWords = stopFile.read().split()
    # print stopWords

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()
feature_names = vectorizer.get_feature_names()
clf =GaussianNB()
clf.fit(X_train, y_train)
scr = clf.score(X_test, y_test)
print scr
