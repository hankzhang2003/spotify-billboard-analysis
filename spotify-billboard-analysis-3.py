import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
from collections import Counter
import ssl
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier)
import tensorflow as tf
from clean_features import clean_features
from clean_weeks import clean_weeks
from make_plots import (make_frequency_plot, make_line_plot, make_dual_plot_same,
                        make_dual_plot_mixed, make_scatter)
from web_scraping import parse_page


features = clean_features()
weeks = clean_weeks()

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


# Web scrape lyrics
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

test = parse_page("Dance the Night Away", "Twice")
test2 = parse_page("7 rings", "Ariana Grande")

from web_scraping import parse_page
allLyrics = {}
for i in range(len(features)):
    print(features['Song'][i], ", ", features['Performer'][i])
    songLyrics = parse_page(features['Song'][i], features['Performer'][i])
    allLyrics[features['SongID'][i]] = songLyrics
    print("Finished row {}".format(i))


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
