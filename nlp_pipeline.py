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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# NLP pipeline to create tokens from lyrics
def lyrics_tokenize(lyrics: str) -> str:
    if lyrics == None or len(lyrics) == 0:
        return []
    
    nfkdForm = unicodedata.normalize("NFKD", lyrics)
    textInput = str(nfkdForm.encode("ASCII", "ignore"))
    sentTokens = sent_tokenize(textInput)
    tokens = list(map(word_tokenize, sentTokens))
    
    stopwords_ = set(stopwords.words('english'))
    punctuation_ = set(string.punctuation)
    tokensFiltered = list(map(lambda s: [w for w in s if not w in stopwords_ and \
                                not w in punctuation_], tokens))
    
    grammar = r"""
        SENT: {<(J|N).*>}
    """
    sent_tags = list(map(pos_tag, tokensFiltered))
    cp = RegexpParser(grammar)
    lyricsTokens = []
    stemmer = SnowballStemmer('english')
    for sent in sent_tags:
        tree = cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'SENT':
                tokenList = [tpos[0].lower() for tpos in subtree.leaves()]
                tokensStemmed = list(map(stemmer.stem, tokenList))
                lyricsTokens.extend(tokensStemmed)
    
    return " ".join(lyricsTokens)
