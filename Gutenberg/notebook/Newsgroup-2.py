
# coding: utf-8

# In[ ]:


import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import utils
from keras.layers import Dense, Input, Flatten, LSTM, GRU
from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalAveragePooling1D
from keras.models import Model, load_model
from nltk import sent_tokenize


# In[22]:


BASE_DIR = '/home/ubuntu/'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
MAX_SEQUENCE_LENGTH = 350
MAX_NB_WORDS = 30000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
GLOVE = False


# In[23]:

if GLOVE:
    print('Indexing word vectors...')
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))


# In[24]:


print('Processing text dataset')

authorlist = []
def load_data(data_type, full_doc=False):
    x = []
    y = []
    with open('../NLP/Gutenberg/Gutenberg/filenames_' +data_type+'.txt', 'r') as filenames:
        for filename in filenames:
            author = filename[:filename.index("_")]
            if author not in authorlist and len(authorlist) <10:
                authorlist.append(author)
            if author not in authorlist:
                continue
            with open('../NLP/Gutenberg/Gutenberg/'+data_type+'/'+ filename.strip(), 'r') as f:
                if full_doc:
                    data = f.read()
                    x.append(data)
                    y.append(authorlist.index(author))
                else:
                    data = sent_tokenize(f.read())
                    x += data
                    y += [authorlist.index(author)] * len(data)
        return [x, y]

texts, labels = load_data('train', full_doc=False)
test_texts, test_labels = load_data('test', full_doc=False)

print('Found %s texts.' % len(texts))
print('Found %s test texts.' % len(test_texts))


# In[25]:


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)


# In[26]:


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[27]:


data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = utils.to_categorical(np.asarray(labels))
test_labels = utils.to_categorical(np.asarray(test_labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# In[28]:


# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]


# In[29]:


if GLOVE:
    print('Preparing embedding matrix.')

    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


# In[30]:


# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
if GLOVE:
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
else:
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH)


# In[ ]:


print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = GRU(256, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(embedded_sequences)
x = GlobalAveragePooling1D()(x)
# x = Conv1D(128, 5, activation='relu')(embedded_sequences)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(35)(x)
# x = Flatten()(x)
# x = Dense(128, activation='relu')(x)
preds = Dense(len(authorlist), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])


# In[ ]:


from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='/tmp/weights2.hdf5', verbose=1, save_best_only=True)

model.fit(x_train, y_train,
          batch_size=500,
          epochs=256,
#           validation_split=0.1,
          validation_data=(x_val, y_val),
          callbacks=[checkpointer])


# model = load_model('/tmp/weights2.hdf5')

# In[ ]:


model.summary()

score, acc = model.evaluate(test, test_labels, batch_size=500)
print('Test score:\n', score)
print('Test accuracy:\n', acc)
# print('Test top_k_accuracy:\n', top_k)


# In[ ]:




