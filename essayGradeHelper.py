import gensim.models.word2vec as word2vec
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.layers import LSTM, Dense, Dropout
from gensim.models import KeyedVectors
import numpy as np
import nltk
import re
from nltk.corpus import stopwords

def essay_to_wordlist(essay_v, remove_stopwords):
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words

def essay_to_sentences(essay_v, remove_stopwords):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    num_words = 0.0
    index2word_set = set(model.index_to_key)
    for word in words:
        if word in index2word_set:
            num_words += 1
            featureVec = np.add(featureVec, model[word])  # Access through `wv`
    if num_words > 0:
        featureVec = np.divide(featureVec, num_words)
    return featureVec

# Function to create feature vectors for a list of essays
def getAvgFeatureVecs(essays, model, num_features):
    essayFeatureVecs = np.zeros((len(essays), num_features), dtype="float32")
    for i, essay in enumerate(essays):
        essayFeatureVecs[i] = makeFeatureVec(essay, model, num_features)
    return essayFeatureVecs


def get_model():
    """Define the model."""
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 300], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()
    return model
