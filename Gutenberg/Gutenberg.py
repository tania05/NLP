#!/usr/bin/env python

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import numpy as np
from nltk.tokenize import sent_tokenize

authorlist = []

def load_data(data_type, full_doc=False):
    x = []
    y = []
    with open('./Gutenberg/filenames_' +data_type+'.txt', 'r') as filenames:
        for filename in filenames:
            author = filename[:filename.index("_")]
            if author not in authorlist:
                authorlist.append(author)
            with open('./Gutenberg/'+data_type+'/'+ filename.strip(), 'r' ) as f:
                if full_doc:
                    data = f.read().decode('iso-8859-1')
                    x.append(data)
                    y.append(authorlist.index(author))
                else:
                    data = sent_tokenize(f.read().decode('iso-8859-1'))
                    x += data
                    y += [authorlist.index(author)] * len(data)

    return [x, y]

data_train, y_train = load_data('train', full_doc=True)
print len(data_train)
print len(y_train)

data_test, y_test = load_data('test', full_doc=True)
print len(data_test)
print len(y_test)

vectorizer = CountVectorizer(encoding='iso-8859-1', stop_words='english',
                            lowercase=True)

X_train = vectorizer.fit_transform(data_train)
X_test = vectorizer.transform(data_test)
feature_names = vectorizer.get_feature_names()
print X_train.shape
print len(feature_names)

# clf = svm.SVC()
# clf = BernoulliNB()
clf = MultinomialNB()
clf.fit(X_train, y_train)
print "fitting done"
y_pred = clf.predict(X_test)
print accuracy_score(y_test, y_pred)