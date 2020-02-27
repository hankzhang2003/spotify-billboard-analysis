# git config --global credential.helper store

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
from collections import Counter
import ssl
import time
import string
import unicodedata

from urllib.request import Request, urlopen
from threading import Thread
from bs4 import BeautifulSoup

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)

import nltk
nltk.download(["stopwords", "punkt", "averaged_perceptron_tagger", "maxent_treebank_pos_tagger"])
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

from clean_features import clean_features
from clean_weeks import clean_weeks
from web_scraping import parse_page, store_lyrics, read_lyrics
from nlp_pipeline import clean_lyrics, lyrics_tokenize
from genre_helper_functions import get_bucket, contains_genre_type, create_genre_column
from make_plots import (make_frequency_plot, make_line_plot, make_dual_plot_same,
                        make_dual_plot_mixed, make_scatter)
import modeling_functions as mf


features = clean_features()
weeks = clean_weeks()

'''
joined = weeks.merge(features, on='SongID')
#joined.to_csv("data/joined.csv", index=False)

# Expand genres into individual components
featureGenres = features.explode('spotify_genre')
featureGenres = featureGenres[featureGenres['spotify_genre'] != '']

joinedGenres = joined.explode('spotify_genre')
joinedGenres = joinedGenres[joinedGenres['spotify_genre'] != '']

explicitness = joined[['Year', 'spotify_track_explicit']]
explicitness = explicitness.groupby(['Year']).mean().reset_index()

numericalMetrics = joined.columns.tolist()[11:23]
numericals = joined[['Year'] + numericalMetrics].groupby(['Year']).mean().reset_index()


# Normalize numerical features not between 0 and 1
featureGenresNorm = featureGenres.copy()
scaled = ["track_duration", "loudness", "tempo"]
for metric in scaled:
    mms = MinMaxScaler()
    featureGenresNorm[metric] = mms.fit_transform(featureGenresNorm['track_duration']. \
                                to_numpy().reshape(-1, 1))

# Create grouped tables
genres = featureGenres.groupby(['spotify_genre'])['SongID'].count().reset_index()
genresJoined = joinedGenres.groupby(['spotify_genre'])['SongID'].count().reset_index()
genresJoinedDecade = joinedGenres.groupby(['spotify_genre', 'Decade'])['SongID'].count(). \
                        reset_index().sort_values(by="Decade")
genreFeatures = featureGenresNorm.groupby(['spotify_genre'])[numericalMetrics].mean().reset_index()
'''


ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

test = parse_page("Dance the Night Away", "Twice")


# Web scrape lyrics
featureScrape = features.loc[[contains_genre_type(genre, ["pop", "rock", "metal"]) for genre \
                             in features['spotify_genre']]].reset_index(drop=True)
lyricsMap = {}
threads = []
temp = 0
start = time.time()
# Write scraped lyrics to hashmap, parallelize to save time (thread safe because no unique keys)
#for i in range(temp, temp+50):
for i in range(len(featureScrape)):
    t = Thread(target=store_lyrics, args=(featureScrape['Song'][i],
                featureScrape['Performer'][i], lyricsMap))
    threads.append(t)
    t.start()
for t in threads:
    t.join()
end = time.time()
print(end - start)
scrapedLyrics = pd.DataFrame(lyricsMap.items(), columns=["SongID", "Lyrics"])
scrapedLyrics.to_csv("data/scrapedLyrics.csv", index=False)


# Get list of all improperly formatted songs and save to file
problemSongs = []
for k, v in lyricsMap.items():
    if v[0][0] == "*":
        problemSongs.append([k] + v[2:5])
print(len(problemSongs))

with open("data/problemSongs.txt", "w") as file:
    for s in problemSongs:
        file.write("{}\n".format(s))


###################

# Read csv of previously outputted scraped lyrics and reformat to match original
allLyrics = read_lyrics()

def valid_lyrics(lyrics: str) -> bool:
    return lyrics[0][0] != "*"
allLyrics = allLyrics[[valid_lyrics(l) for l in allLyrics['Lyrics']]]
allLyrics['Lyrics'] = allLyrics['Lyrics'].map(clean_lyrics)
allLyrics.reset_index(drop=True, inplace=True)

testpage = parse_page("Dance the Night Away", "Twice")
testpage = [line.replace(",", "") for line in testpage]
testpage = clean_lyrics(testpage)
print(testpage)
testpage_tokenized = lyrics_tokenize(testpage)
print(testpage_tokenized)
testpage2 = parse_page("Fake Love", "BTS")
testpage2 = [line.replace(",", "") for line in testpage2]
testpage2 = clean_lyrics(testpage2)
testpage2_tokenized = lyrics_tokenize(testpage2)


# NLP pipeline to create tokens -> bag of words -> corpus
start = time.time()
allLyrics['Lyrics_tokenized'] = list(map(lyrics_tokenize, allLyrics['Lyrics']))
allLyrics.to_csv("data/allLyricsTokenized.csv", index=False)


# TF-IDF
corpus = [testpage_tokenized, testpage2_tokenized]
tf = CountVectorizer()
tf_matrix = tf.fit_transform(corpus)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(corpus)










model = Sequential()
