import numpy as np
import pandas as pd
import string
import unicodedata
import nltk
nltk.download(["stopwords", "punkt", "averaged_perceptron_tagger", "maxent_treebank_pos_tagger"])
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk import RegexpParser
from nltk.stem.snowball import SnowballStemmer


def clean_lyrics(lyrics: list) -> str:
    cleanedLyrics = [line for line in lyrics if len(line) != 0 and line[0] != "["]
    return " ".join(cleanedLyrics)

# NLP pipeline
def lyrics_tokenize(lyrics: str) -> str:
    if lyrics == None or len(lyrics) == 0:
        return []
    
    nfkd_form = unicodedata.normalize("NFKD", lyrics)
    text_input = str(nfkd_form.encode("ASCII", "ignore"))
    sent_tokens = sent_tokenize(text_input)
    tokens = list(map(word_tokenize, sent_tokens))
    
    stopwords_ = set(stopwords.words('english'))
    punctuation_ = set(string.punctuation)
    tokens_filtered = list(map(lambda s: [w for w in s if not w in stopwords_ and not w in punctuation_], tokens))
    
    sent_tags = list(map(pos_tag, tokens_filtered))

    grammar = r"""
        SENT: {<(J|N).*>}                # chunk sequences of proper nouns
    """
    cp = RegexpParser(grammar)
    lyric_tokens = []
    stemmer_snowball = SnowballStemmer('english')
    for sent in sent_tags:
        tree = cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == "SENT":
                tokenlist = [tpos[0].lower() for tpos in subtree.leaves()]
                tokens_stemsnowball = list(map(stemmer_snowball.stem, tokenlist))
                lyric_tokens.extend(tokens_stemsnowball)

    return lyric_tokens
