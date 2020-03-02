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
#%matplotlib inline

from urllib.request import Request, urlopen
from threading import Thread
from bs4 import BeautifulSoup

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                            GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.naive_bayes import MultinomialNB

import nltk
nltk.download(["stopwords", "punkt", "averaged_perceptron_tagger", "maxent_treebank_pos_tagger"])
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

from clean_dfs import clean_features, clean_weeks, clean_lyrics
from web_scraping import parse_page, store_lyrics
from nlp_pipeline import lyrics_tokenize, get_tfidf_matrix
from genre_helper_functions import get_bucket, contains_genre_type, create_genre_column
from make_plots import (make_frequency_plot, make_line_plot, make_dual_plot_same,
                        make_dual_plot_mixed, make_scatter)
import modeling_functions as mf


features = clean_features()
weeks = clean_weeks()


'''joined = weeks.merge(features, on='SongID')
#joined.to_csv("data/joined.csv", index=False)

# Expand genres into individual components
featureGenres = features.explode('spotify_genre')
featureGenres = featureGenres[featureGenres['spotify_genre'] != '']

joinedGenres = joined.explode('spotify_genre')
joinedGenres = joinedGenres[joinedGenres['spotify_genre'] != '']

explicitness = joined[['Year', 'spotify_track_explicit']]
explicitness = explicitness.groupby(['Year']).mean().reset_index()

numericalMetrics = joined.columns.tolist()[11:23]
numericals = joined[['Year'] + numericalMetrics].groupby(['Year']).mean().reset_index()'''


'''# Normalize numerical features not between 0 and 1
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
genreFeatures = featureGenresNorm.groupby(['spotify_genre'])[numericalMetrics].mean().reset_index()'''


ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


'''# Web scrape lyrics
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
        file.write("{}\n".format(s))'''

# Read csv of previously outputted scraped lyrics and reformat to match original
# allLyrics = clean_lyrics()

'''# NLP pipeline to create tokens from lyrics
allLyrics['Lyrics_tokenized'] = list(map(lyrics_tokenize, allLyrics['Lyrics']))
allLyrics.dropna(inplace=True)
allLyrics.to_csv("data/lyricsTokenized.csv", index=False)'''


###################

# Create corpus and make dataframe with TF-IDF matrix
allLyrics = pd.read_csv("data/lyricsTokenized.csv")
allLyrics.dropna(inplace=True)
corpus = allLyrics['Lyrics_tokenized']
tfidfLyrics = get_tfidf_matrix(corpus, 5000)
tfidfLyrics.insert(0, "SongID", allLyrics['SongID'])
tfidfLyrics.to_csv("data/tfidfMatrix.csv", index=False)


# Run models with only lyrics, not counting other features

# Join with valence column from features to get valence of each song
valenceOnly = pd.DataFrame({"SongID": features['SongID'], "spotify_genre": 
                            features['spotify_genre'], "valence": features['valence']})
lyricsAndValence = tfidfLyrics.merge(valenceOnly, on='SongID')
lyricsAndValence.set_index("SongID", inplace=True)
# Create new dataframe using classifier instead of regressor
lyricsAndValenceBin = lyricsAndValence.copy()
lyricsAndValenceBin['valence'] = (lyricsAndValenceBin['valence'] > 0.5).astype(int)


# Run classifier models for pop genre
lyricsAndValenceBinPop = lyricsAndValenceBin[[contains_genre_type(g, ["pop"]) \
                                for g in lyricsAndValenceBin['spotify_genre']]]
lyricsAndValenceBinPop.drop(["spotify_genre"], axis=1, inplace=True)
X = lyricsAndValenceBinPop[lyricsAndValenceBinPop.columns.difference(['valence'])]
y = lyricsAndValenceBinPop['valence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Logistic regression model
logistic_regression_results = mf.get_logistic_regression_results(X_train, \
                                        X_test, y_train, y_test)
print(logistic_regression_results)
# 0.5559, 0.3957, 0.3917

# Explore gradient boosting hyperparameters
'''start = time.time()
mf.plot_gradient_boost_class_hyperparameters(X_train, X_test, y_train, y_test, \
                                             "pop")
end = time.time()
print(end-start)'''

# Gradient boosting classifier model
gradient_boost_class_results = mf.get_gradient_boost_class_results(0.1, 140, 3, \
                                        X_train, X_test, y_train, y_test)
print(gradient_boost_class_results)
# 0.6347, 0.0585, 0.4894


# Run regressor models for pop genre
lyricsAndValencePop = lyricsAndValence[[contains_genre_type(g, ["pop"]) \
                            for g in lyricsAndValence['spotify_genre']]]
lyricsAndValencePop.drop(["spotify_genre"], axis=1, inplace=True)
X = lyricsAndValencePop[lyricsAndValencePop.columns.difference(['valence'])]
y = lyricsAndValencePop['valence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Grid search gradient boosting regressor hyperparameters and return model score and RMSE
gbr = mf.grid_search_gradient_boost(X_train, X_test, y_train, y_test)
print(gbr.best_params_, np.sqrt(np.abs(gbr.best_score_)))
scoreValencePop = gbr.score(X_test, y_test)
y_pred = gbr.predict(X_test)
rmseValencePop = np.sqrt(mean_squared_error(y_test, y_pred))
print(scoreValencePop, rmseValencePop)


# Run classifier models for rock/metal genres
lyricsAndValenceBinRock = lyricsAndValenceBin[[contains_genre_type(g, ["rock", "metal"]) \
                                for g in lyricsAndValenceBin['spotify_genre']]]
lyricsAndValenceBinRock.drop(["spotify_genre"], axis=1, inplace=True)
X = lyricsAndValenceBinRock[lyricsAndValenceBinRock.columns.difference(['valence'])]
y = lyricsAndValenceBinRock['valence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Logistic regression model
logistic_regression_results = mf.get_logistic_regression_results(X_train, \
                                        X_test, y_train, y_test)
print(logistic_regression_results)
# 0.5877, 0.4042, 0.3867

# Explore gradient boosting hyperparameters
'''start = time.time()
mf.plot_gradient_boost_class_hyperparameters(X_train, X_test, y_train, y_test, \
                                             "pop")
end = time.time()
print(end-start)'''

# Gradient boosting classifier model
gradient_boost_class_results = mf.get_gradient_boost_class_results(0.1, 140, 3, \
                                        X_train, X_test, y_train, y_test)
print(gradient_boost_class_results)
# 0.6644, 0.0436, 0.4630


# Run regressor models for rock/metal genres
lyricsAndValenceRock = lyricsAndValence[[contains_genre_type(g, ["rock", "metal"]) \
                            for g in lyricsAndValence['spotify_genre']]]
lyricsAndValenceRock.drop(["spotify_genre"], axis=1, inplace=True)
X = lyricsAndValenceRock[lyricsAndValenceRock.columns.difference(['valence'])]
y = lyricsAndValenceRock['valence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Grid search gradient boosting regressor hyperparameters and return model score and RMSE
gbr = mf.grid_search_gradient_boost(X_train, X_test, y_train, y_test)
print(gbr.best_params_, np.sqrt(np.abs(gbr.best_score_)))
scoreValenceRock = gbr.score(X_test, y_test)
y_pred = gbr.predict(X_test)
rmseValenceRock = np.sqrt(mean_squared_error(y_test, y_pred))
print(scoreValenceRock, rmseValenceRock)


# Now try adding all other numerical features to see if it improves accuracy

# Join with features to get all numerical features as well as valence
lyricsAndFeatures = tfidfLyrics.merge(features, on='SongID')
lyricsAndFeatures.drop(["Performer", "Song"], axis=1, inplace=True)
lyricsAndFeatures.set_index("SongID", inplace=True)
# Create new dataframe using classifier instead of regressor
lyricsAndFeaturesBin = lyricsAndFeatures.copy()
lyricsAndFeaturesBin['valence'] = (lyricsAndFeaturesBin['valence'] > 0.5).astype(int)


# Run classifier models for pop genre
lyricsAndFeaturesBinPop = lyricsAndFeaturesBin[[contains_genre_type(g, ["pop"]) \
                                for g in lyricsAndFeaturesBin['spotify_genre']]]
lyricsAndFeaturesBinPop.drop(["spotify_genre"], axis=1, inplace=True)
X = lyricsAndFeaturesBinPop[lyricsAndFeaturesBinPop.columns.difference(['valence'])]
y = lyricsAndFeaturesBinPop['valence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Logistic regression model
logistic_regression_results = mf.get_logistic_regression_results(X_train, \
                                        X_test, y_train, y_test)
print(logistic_regression_results)
# 0.6796, 0.5280, 0.5646

# Explore gradient boosting hyperparameters
'''start = time.time()
mf.plot_gradient_boost_class_hyperparameters(X_train, X_test, y_train, y_test, \
                                             "pop")
end = time.time()
print(end-start)'''

# Gradient boosting classifier model
gradient_boost_class_results = mf.get_gradient_boost_class_results(0.1, 140, 3, \
                                        X_train, X_test, y_train, y_test)
print(gradient_boost_class_results)
# 0.7821, 0.5954, 0.7548


# Run regressor models for pop genre
lyricsAndFeaturesPop = lyricsAndFeatures[[contains_genre_type(g, ["pop"]) \
                            for g in lyricsAndFeatures['spotify_genre']]]
lyricsAndFeaturesPop.drop(["spotify_genre"], axis=1, inplace=True)
X = lyricsAndFeaturesPop[lyricsAndFeaturesPop.columns.difference(['valence'])]
y = lyricsAndFeaturesPop['valence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Grid search gradient boosting regressor hyperparameters and return model score and RMSE
gbr = mf.grid_search_gradient_boost(X_train, X_test, y_train, y_test)
print(gbr.best_params_, np.sqrt(np.abs(gbr.best_score_)))
scoreFeaturesPop = gbr.score(X_test, y_test)
y_pred = gbr.predict(X_test)
rmseFeaturesPop = np.sqrt(mean_squared_error(y_test, y_pred))
print(scoreFeaturesPop, rmseFeaturesPop)


# Run classifier models for rock/metal genres
lyricsAndFeaturesBinRock = lyricsAndFeaturesBin[[contains_genre_type(g, ["rock", "metal"]) \
                                for g in lyricsAndFeaturesBin['spotify_genre']]]
lyricsAndFeaturesBinRock.drop(["spotify_genre"], axis=1, inplace=True)
X = lyricsAndFeaturesBinRock[lyricsAndFeaturesBinRock.columns.difference(['valence'])]
y = lyricsAndFeaturesBinRock['valence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Logistic regression model
logistic_regression_results = mf.get_logistic_regression_results(X_train, \
                                        X_test, y_train, y_test)
print(logistic_regression_results)
# 0.7294, 0.6028, 0.5925

# Explore gradient boosting hyperparameters
'''start = time.time()
mf.plot_gradient_boost_class_hyperparameters(X_train, X_test, y_train, y_test, \
                                             "pop")
end = time.time()
print(end-start)'''

# Gradient boosting classifier model
gradient_boost_class_results = mf.get_gradient_boost_class_results(0.1, 140, 3, \
                                        X_train, X_test, y_train, y_test)
print(gradient_boost_class_results)
# 0.8055, 0.6045, 0.7626


# Run regressor models for rock/metal genres
lyricsAndFeaturesRock = lyricsAndFeatures[[contains_genre_type(g, ["rock", "metal"]) \
                            for g in lyricsAndFeatures['spotify_genre']]]
lyricsAndFeaturesRock.drop(["spotify_genre"], axis=1, inplace=True)
X = lyricsAndFeaturesRock[lyricsAndFeaturesRock.columns.difference(['valence'])]
y = lyricsAndFeaturesRock['valence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Grid search gradient boosting regressor hyperparameters and return model score and RMSE
gbr = mf.grid_search_gradient_boost(X_train, X_test, y_train, y_test)
print(gbr.best_params_, np.sqrt(np.abs(gbr.best_score_)))
scoreFeaturesRock = gbr.score(X_test, y_test)
y_pred = gbr.predict(X_test)
rmseFeaturesRock = np.sqrt(mean_squared_error(y_test, y_pred))
print(scoreFeaturesRock, rmseFeaturesRock)