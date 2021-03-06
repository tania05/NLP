#!/usr/bin/env python
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB

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

def regex_tokenizer(doc):
    """Return a function that split a string in sequence of tokens"""
    return doc.split(' ')

vectorizer = CountVectorizer(lowercase=False, stop_words=None,  max_df=1.0, min_df=1, max_features=None, tokenizer=regex_tokenizer )

X_train = vectorizer.fit_transform(X_train).toarray()
print X_train.shape
X_test = vectorizer.transform(X_test).toarray()
feature_names = vectorizer.get_feature_names()
clf = MultinomialNB()
clf.fit(X_train, y_train)
scr = clf.score(X_test, y_test)
print scr


feature_names = vectorizer.get_feature_names()
for i, class_label in enumerate(clf.classes_):
    top10 = np.argsort(clf.coef_[i])[-30:]
    print("%s: %s" % (class_label,
	  " ".join(feature_names[j] for j in top10)))