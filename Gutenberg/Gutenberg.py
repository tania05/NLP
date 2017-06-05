#!/usr/bin/env python

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import numpy as np
from nltk.tokenize import sent_tokenize

data_train = []
y_train = []

X_test = []
y_test = []

authorlist = []

with open('./Gutenberg/filenames.txt', 'r') as filenames:
    for filename in filenames:
        author = filename[:filename.index("_")]
        if author not in authorlist:
            authorlist.append(author)
        with open('./Gutenberg/txt/' + filename.strip(), 'r' ) as f:
            data = sent_tokenize(f.read().decode('iso-8859-1'))
            # data = f.read().decode('iso-8859-1').encode('utf-8').splitlines()
            data_train += data
            y_train += [authorlist.index(author)] * len(data)

print len(data_train)
print len(y_train)

# vectorizer = TfidfVectorizer()
vectorizer = CountVectorizer(encoding='iso-8859-1', stop_words='english',
                            lowercase=True, binary=True)

X_train = vectorizer.fit_transform(data_train).toarray()
X_test = vectorizer.transform(X_test).toarray()
feature_names = vectorizer.get_feature_names()
print X_train.shape
print len(feature_names)

# clf = svm.SVC()
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print accuracy_score(y_test, y_pred)