#!/usr/bin/env python

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

from itertools import chain
import shutil
import os

m_authorList = {}
with open('filenames.txt', 'r') as f:
    for filename in f:
        m_author = filename[:filename.index("_")]
        m_authorList[m_author] = m_authorList.get(m_author, 0) +1

median = np.median(sorted(m_authorList.values(), reverse=True))

print "Original # of books ", sum(m_authorList.values())
print "Original # of authors ",len(m_authorList.keys())

authorList = {}
with open('filenames.txt', 'r') as f:
    for filename in f:
        author = filename[:filename.index("_")]
        lst = authorList.get(author, [])
        lst.append(filename.strip())
        authorList[author] = lst

train = map(lambda lst: lst[:int(0.7 * len(lst))] if len(lst) >= median else [], authorList.values())
test = map(lambda lst: lst[int(0.7 * len(lst)):] if len(lst) >= median else [], authorList.values())

train = filter(lambda s: len(s) > 0, list(chain.from_iterable(train)))
test = filter(lambda s: len(s) > 0, list(chain.from_iterable(test)))

print "# total books greater than median per author ",(len(train) + len(test))

remainingAuthors = [k for k, v in m_authorList.items() if v >= median]

print "# of remaining authors having books greater than median ",len(remainingAuthors)

#for e in test:
    #base = "/home/ubuntu/NLP/Gutenberg/Gutenberg/"
 #   src = 'txt/' + e
  #  dst = 'normalizedTest/' + e
   # shutil.copyfile(src, dst)

#for e in train:
    #base = "/home/ubuntu/NLP/Gutenberg/Gutenberg/"
 #   src = 'txt/' + e
  #  dst = 'normalizedTrain/' + e
   # shutil.copyfile(src, dst)
