import numpy as np
import pandas as pd
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


def clean_lyrics(lyrics: list) -> str:
    cleanedLyrics = [line for line in lyrics if len(line) != 0 and line[0] != "["]
    return " ".join(cleanedLyrics)

def lyrics_tokenize(lyrics: str) -> str:
    return tokens
