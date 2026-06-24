from __future__ import annotations

import re
import string
from typing import Iterable

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for resource in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
HEADER_PATTERNS = [
    r"from:.*",
    r"to:.*",
    r"subject:.*",
    r"cc:.*",
    r"bcc:.*",
    r"date:.*",
    r"reply-to:.*",
]


def strip_html(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    return soup.get_text(separator=" ")


def remove_headers(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        low = line.strip().lower()
        if any(re.match(pattern, low) for pattern in HEADER_PATTERNS):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def normalize_text(text: str) -> str:
    text = str(text)
    text = strip_html(text)
    text = remove_headers(text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", text)
    text = re.sub(r"\S+@\S+", " EMAILTOKEN ", text)
    text = re.sub(r"\d+", " NUMTOKEN ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> list[str]:
    tokens = word_tokenize(text)
    clean_tokens = []
    for token in tokens:
        token = token.strip(string.punctuation)
        if not token or token in STOPWORDS or len(token) <= 1:
            continue
        lemma = LEMMATIZER.lemmatize(token)
        clean_tokens.append(lemma)
    return clean_tokens


def preprocess_text(text: str) -> str:
    normalized = normalize_text(text)
    tokens = tokenize_and_lemmatize(normalized)
    return " ".join(tokens)


def preprocess_corpus(texts: Iterable[str]) -> list[str]:
    return [preprocess_text(text) for text in texts]
