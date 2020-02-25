import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
from collections import Counter
import string
import ssl
import time
from urllib.request import Request, urlopen
from threading import Thread
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from clean_features import clean_features
from clean_weeks import clean_weeks
from web_scraping import parse_page, store_lyrics
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

# Web scrape lyrics
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


from web_scraping import parse_page, store_lyrics
test = parse_page("Dance the Night Away", "Twice")

featureScrape= features.loc[[contains_genre_type(genre, ["pop", "rock", "metal"]) for genre \
                            in features['spotify_genre']]].reset_index(drop=True)
allLyrics = {}
threads = []
temp = 0
start = time.time()
for i in range(temp, temp+1000):
#for i in range(len(featureScrape)):
    t = Thread(target=store_lyrics, args=(featureScrape['Song'][i], featureScrape['Performer'][i], allLyrics))
    threads.append(t)
    t.start()
for t in threads:
    t.join()
end = time.time()
print(end - start)
rockLyrics = pd.DataFrame(allLyrics.items(), columns=["SongID", "Lyrics"])
rockLyrics.to_csv("data/scrapedLyrics.csv", index=False)

problemSongs = []
for k, v in allLyrics.items():
    if v[0][0] == "*":
        problemSongs.append([k] + v[2:5])
print(len(problemSongs))

with open("data/problemSongs.txt", "w") as file:
    for s in problemSongs:
        file.write("{}\n".format(s))


model = Sequential()
