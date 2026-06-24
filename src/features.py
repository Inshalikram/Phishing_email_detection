from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from .config import CONFIG


@dataclass
class SequenceArtifacts:
    tokenizer: Tokenizer
    X_train: any
    X_val: any
    X_test: any


def build_bow(train_texts, val_texts, test_texts):
    vectorizer = CountVectorizer(max_features=CONFIG.max_features)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_val, X_test


def build_tfidf(train_texts, val_texts, test_texts):
    vectorizer = TfidfVectorizer(max_features=CONFIG.max_features, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_val, X_test


def build_sequences(train_texts, val_texts, test_texts):
    tokenizer = Tokenizer(num_words=CONFIG.max_features, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=CONFIG.max_len, padding="post")
    X_val = pad_sequences(tokenizer.texts_to_sequences(val_texts), maxlen=CONFIG.max_len, padding="post")
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=CONFIG.max_len, padding="post")
    return SequenceArtifacts(tokenizer, X_train, X_val, X_test)
