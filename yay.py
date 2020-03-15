from __future__ import absolute_import, division, print_function, unicode_literals

"""This code is used to read all news and their labels"""
import os
import glob
import nltk
import collections
import matplotlib.pyplot as plt
import numpy as np
import gc

import tensorflow as tf
import tensorflow.keras as keras
from keras.utils import to_categorical

# import keras
import pdb
from tensorflow.keras.layers import Layer, RNN, Dense, Embedding
from tensorflow.keras.models import Sequential

from utils import *

# nltk.download('punkt')
def to_categories(name, cat=["politics","rec","comp","religion"]):
    for i in range(len(cat)):
        if str.find(name,cat[i])>-1:
            return(i)
    print("Unexpected folder: " + name) # print the folder name which does not include expected categories
    return("wth")

def data_loader(images_dir):
    categories = os.listdir(data_path)
    news = [] # news content
    groups = [] # category which it belong to
    
    for cat in categories:
        print("Category:"+cat)
        for the_new_path in glob.glob(data_path + '/' + cat + '/*'):
            news.append(open(the_new_path,encoding = "ISO-8859-1", mode ='r').read())
            groups.append(cat)

    return news, list(map(to_categories, groups))

CELL_OUTPUTS = 500

class MinimalRNNCell(Layer):
    def __init__(self, units, embed_size, num_words, **kwargs):
        self.state_size = units
        self.embed_size = embed_size
        self.num_words = num_words
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel_1 = self.add_weight(shape=(self.state_size + self.embed_size, self.state_size), initializer='uniform', name='kernel_1')
        self.kernel_2 = self.add_weight(shape=(self.state_size, self.num_words), initializer='uniform', name='kernel_2')

        self.built = True

    # return y_t, s_(t+1)
    # or output, next_hidden_state
    def call(self, current_input, current_states):
        # inputs is same size as input_shape in "build"
        # basically turns a sentence into another sentence (X * embed_len => X * num_words)
        # print("Input is", inputs)
        current_state = current_states[0]
        concat = tf.concat([current_state, current_input], axis=1)

        logits = tf.matmul(concat, self.kernel_1)
        next_hidden_state = tf.sigmoid(logits)
        output = tf.nn.softmax(tf.matmul(next_hidden_state, self.kernel_2))

        return output, [next_hidden_state]

(EMBED_SIZE, NUM_WORDS, tokens, sentences, word_to_idx, idx_to_word, embedding_matrix) = readShitIn()
as_index = convert_to_idx(tokens, word_to_idx)
X_train, X_valid, Y_train, Y_valid = create_train_valid(as_index, NUM_WORDS)

def X_gen():
    for idx in range(len(X_train)):
        X = X_train[idx].reshape(1, 1)
        Y = Y_train[idx].reshape(1, NUM_WORDS)
        
        # pdb.set_trace()

        yield X, Y

sentences_as_index = convert_sentences_idx(sentences, word_to_idx)
train_generator, valid_generator = create_train_valid_sentences(sentences_as_index, NUM_WORDS)

pdb.set_trace()
# setup the RNN
cells = [MinimalRNNCell(CELL_OUTPUTS, EMBED_SIZE, NUM_WORDS)]
rnn = RNN(cells, return_sequences=False, input_shape=(None, 1))

model = Sequential()
model.add(Embedding(input_dim=NUM_WORDS, output_dim=EMBED_SIZE, weights=[embedding_matrix], trainable=False))
model.add(rnn)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_freq=5,
                                                 verbose=1)

history = model.fit(
    X_train,
    Y_train,
    batch_size=1,
    epochs=1,
    callbacks=[cp_callback],
    verbose=True)

# history = model.fit_generator(
#     X_gen(),
#     steps_per_epoch=40,
#     epochs=100,
#     verbose=True,
#     callbacks=[cp_callback]
# )

loss, acc = model.evaluate(X_valid, Y_valid, verbose=2)
print("accuracy on validation set: {:5.2f}%".format(100*acc))

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
loss, acc = model.evaluate(X_train, Y_train, verbose=2)
print("train accuracy: {:5.2f}%".format(100*acc))
loss, acc = model.evaluate(X_valid, Y_valid, verbose=2)
print("validation accuracy: {:5.2f}%".format(100*acc))

# SENTENCES START
# checkpoint_sentences_path = "cp-sentences/cp-{epoch:04d}.ckpt"
# checkpoint_sentences_dir = os.path.dirname(checkpoint_sentences_path)

# # Create a callback that saves the model's weights
# cp_callback_2 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_sentences_path,
#                                                  save_weights_only=True,
#                                                  save_freq=5,
#                                                  verbose=1)

# history = model.fit_generator(
#     train_generator(),
#     steps_per_epoch=10,
#     epochs=100,
#     verbose=True)

# loss, acc = model.evaluate(X_train, Y_train, verbose=2)
# print("train accuracy: {:5.2f}%".format(100*acc))
# loss, acc = model.evaluate(X_valid, Y_valid, verbose=2)
# print("validation accuracy: {:5.2f}%".format(100*acc))

pdb.set_trace()
# news, groups = data_loader(data_path)