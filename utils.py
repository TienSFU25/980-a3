import numpy as np
import nltk
import pdb
import gc

from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle

data_path = "datasets/20news_subsampled"
news_path = "datasets/eng_lit.txt"
TRAIN_FRACTION = 0.7
RANDOM_STATE = 50
THRESHOLD = 3
UNK = "__UNK__"

def readShitIn():
    news_data = ''

    with open(news_path) as f:
        news_data = f.read()

    tokens = nltk.word_tokenize(news_data)
    whole_sentences = nltk.tokenize.sent_tokenize(news_data)
    sentences = [nltk.word_tokenize(whole_sentence) for whole_sentence in whole_sentences]

    word_counter = {}

    for token in tokens:
        if token in word_counter:
            word_counter[token] += 1
        else:
            word_counter[token] = 0

    eligible_words = []
    for word, count in word_counter.items():
        if count > THRESHOLD:
            eligible_words.append(word)

    word_to_idx = {}
    idx_to_word = {}

    counter = 0
    for token in eligible_words:
        if token not in word_to_idx:
            word_to_idx[token] = counter
            idx_to_word[counter] = token
            counter += 1

    NUM_WORDS = len(word_to_idx) + 1
    idx_to_word[len(idx_to_word)] = UNK

    # read in the glove shit
    glove_vectors = 'datasets/glove.6B.100d.txt'
    glove = np.loadtxt(glove_vectors, dtype='str', comments=None)

    vectors = glove[:, 1:].astype('float')
    words = glove[:, 0]
    EMBED_SIZE = vectors.shape[1]

    del glove

    word_lookup = {word: vector for word, vector in zip(words, vectors)}
    embedding_matrix = np.zeros((NUM_WORDS, EMBED_SIZE))
    not_found = 0

    for i, word in enumerate(word_to_idx.keys()):
        # pdb.set_trace()
        # Look up the word embedding
        vector = word_lookup.get(word, None)

        # Record in matrix
        if vector is not None:
            embedding_matrix[i] = vector
        else:
            not_found += 1

    print("Could not find", not_found, "embeddings")

    embedding_matrix = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1).reshape((-1, 1))
    embedding_matrix = np.nan_to_num(embedding_matrix)

    return (EMBED_SIZE, NUM_WORDS, tokens, sentences, word_to_idx, idx_to_word, embedding_matrix)

def convert_to_idx(words, word_to_idx):
    unk_index = len(word_to_idx)
    return [(word_to_idx[word] if word in word_to_idx else unk_index) for word in words]

def convert_sentences_idx(sentences, word_to_idx):
    unk_index = len(word_to_idx)
    return [[(word_to_idx[word] if word in word_to_idx else unk_index) for word in sentence] for sentence in sentences]

def create_train_valid_sentences(sentences,
                                 NUM_WORDS,
                                 top=100,
                                 train_fraction=TRAIN_FRACTION):

    features = []
    labels = []

    max_sentence_length = -1
    for sentence in sentences:
        max_sentence_length = max(max_sentence_length, len(sentence))

    i = 0

    for sentence in sentences:
        features.append(sentence)
        next_idx = 0

        if i < len(sentences) - 1:
            next_sentence = sentences[i + 1]

            if len(next_sentence) > 0:
                next_idx = next_sentence[0]

        pred = sentence[1:]
        pred.append(next_idx)
        
        labels.append(pred)
        i += 1

    # pdb.set_trace()
    # 12640 * 248
    # features = pad_sequences(features, maxlen=max_sentence_length, padding='post')
    # labels = pad_sequences(labels, maxlen=max_sentence_length, padding='post')

    # Randomly shuffle features and labels
    features, labels = shuffle(features, labels, random_state=RANDOM_STATE)

    # Decide on number of samples for training
    train_end = int(train_fraction * len(labels))

    X_train = np.array(features[:train_end])
    X_valid = np.array(features[train_end:])

    train_labels = labels[:train_end]
    valid_labels = labels[train_end:]

    def train_generator():
        for idx in range(len(train_labels)):
            # print("Generated idx is", idx)
            x_train = X_train[idx]
            x_train = np.array(x_train).reshape((len(x_train), 1))
            y_train = np.zeros((len(x_train), NUM_WORDS))

            for index, word_idx in enumerate(x_train):
                y_train[index, word_idx] = 1

            # pdb.set_trace()
            yield x_train, y_train

    def valid_generator():
        for idx in range(len(valid_labels)):
            x = X_valid[idx]
            x = np.array(x).reshape((len(x), 1))
            y = np.zeros((len(x), NUM_WORDS))

            for index, word_idx in enumerate(x):
                y[index, word_idx] = 1

            yield x, y

    # pdb.set_trace()

    # Convert to arrays
    # X_train = np.array(train_features).reshape((len(train_labels), 1))
    # X_valid = np.array(valid_features).reshape((len(valid_labels), 1))

    # num_sentences * words/sentence * embeddings/word
    # y_train = np.zeros((len(train_labels), max_sentence_length, NUM_WORDS), dtype=np.int8)
    # y_valid = np.zeros((len(valid_labels), max_sentence_length, NUM_WORDS), dtype=np.int8)
    # pdb.set_trace()

    # One hot encoding of labels
    # for example_index, word_index in enumerate(train_labels):
    #     y_train[example_index, word_index] = 1

    # for example_index, word_index in enumerate(valid_labels):
    #     y_valid[example_index, word_index] = 1

    # Memory management
    gc.enable()
    # del features, labels, train_labels, valid_labels
    gc.collect()

    return train_generator, valid_generator

def create_train_valid(tokens,
                       NUM_WORDS,
                       top=-1,
                       train_fraction=TRAIN_FRACTION):
    """Create training and validation features and labels."""

    if top == -1:
        top = len(tokens)
        features = tokens[:top]
        labels = tokens[1:top+1]
        labels.append(tokens[0])
    else:
        features = tokens[:top]
        labels = tokens[1:top+1]        

    # Randomly shuffle features and labels
    features, labels = shuffle(features, labels, random_state=RANDOM_STATE)

    # Decide on number of samples for training
    train_end = int(train_fraction * len(labels))

    train_features = np.array(features[:train_end])
    valid_features = np.array(features[train_end:])

    train_labels = labels[:train_end]
    valid_labels = labels[train_end:]

    # Convert to arrays
    X_train = np.array(train_features).reshape((len(train_labels), 1))
    X_valid = np.array(valid_features).reshape((len(valid_labels), 1))

    # Using int8 for memory savings
    y_train = np.zeros((len(train_labels), NUM_WORDS), dtype=np.int8)
    y_valid = np.zeros((len(valid_labels), NUM_WORDS), dtype=np.int8)
    # pdb.set_trace()

    # One hot encoding of labels
    for example_index, word_index in enumerate(train_labels):
        y_train[example_index, word_index] = 1

    for example_index, word_index in enumerate(valid_labels):
        y_valid[example_index, word_index] = 1

    # Memory management
    gc.enable()
    del features, labels, train_features, valid_features, train_labels, valid_labels
    gc.collect()

    return X_train, X_valid, y_train, y_valid
