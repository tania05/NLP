#!/usr/bin/env python

from itertools import chain
import shutil
import os

authorList = {}
with open('filenames_test.txt', 'r') as f:
    for filename in f:
        author = filename[:filename.index("_")]
        lst = authorList.get(author, [])
        lst.append(filename.strip())
        authorList[author] = lst


for author, lst in authorList.items():
    for e in lst[4:]:
        os.remove("/home/ubuntu/NLP/Gutenberg/Gutenberg/test/"+e)

# train = map(lambda lst: lst[:int(0.7 * len(lst))], authorList.values())
# test = map(lambda lst: lst[int(0.7 * len(lst)):], authorList.values())

# train = filter(lambda s: len(s) > 0, list(chain.from_iterable(train)))
# test = filter(lambda s: len(s) > 0, list(chain.from_iterable(test)))


# for e in test:
#     base = "/home/ubuntu/NLP/Gutenberg/Gutenberg/"
#     src = base +'txt/' + e
#     dst = base +'test/' + e
#     shutil.copyfile(src, dst)

# for e in train:
#     base = "/home/ubuntu/NLP/Gutenberg/Gutenberg/"
#     src = base +'txt/' + e
#     dst = base +'train/' + e
#     shutil.copyfile(src, dst)
