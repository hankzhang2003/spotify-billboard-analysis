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
    
    grammar = r"""
        SENT: {<(J|N).*>}
    """
    sent_tags = list(map(pos_tag, tokens))
    cp = RegexpParser(grammar)
    regexTokens = []
    stemmer = SnowballStemmer("english")
    for sent in sent_tags:
        tree = cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'SENT':
                tokenList = [tpos[0].lower() for tpos in subtree.leaves()]
                tokensStemmed = list(map(stemmer.stem, tokenList))
                regexTokens.extend(tokensStemmed)
    
    tokensFiltered = regexTokens[1:]
    stopwords_ = set(stopwords.words("english"))
    tokensFiltered = [w for w in tokensFiltered if not w in stopwords_]
    tokenString = " ".join(tokensFiltered)
    punctuation_ = set(string.punctuation)
    for p in punctuation_:
        tokenString = tokenString.replace(p, "")
    tokenString = tokenString.replace("u2005", " ")
    return tokenString

def get_tfidf_matrix(corpus: pd.Series, features: int) -> pd.DataFrame:
    tfidf = TfidfVectorizer(max_features=features)
    tfidfMatrix = tfidf.fit_transform(corpus)
    tfidfDF = pd.DataFrame(tfidfMatrix.toarray(), columns=tfidf.get_feature_names())
    return tfidfDF
