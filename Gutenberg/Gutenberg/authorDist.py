#!/usr/bin/env python

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

authorList = {}
with open('filenames.txt', 'r') as f:
    for filename in f:
        author = filename[:filename.index("_")]
        authorList[author] = authorList.get(author, 0) +1
       

authors = sorted(authorList, key=authorList.get, reverse=True)
count = sorted(authorList.values(), reverse=True)
 
y_pos = np.arange(len(authors))

 
plt.bar(y_pos, count, align='edge', alpha=0.5)
plt.xticks(y_pos, authors, rotation = 'vertical')
plt.ylabel('Count')
plt.title('Author distribution')
 
plt.show()

