import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                            GradientBoostingClassifier, GradientBoostingRegressor)
from clean_features import clean_features
from clean_weeks import clean_weeks
from genre_helper_functions import get_bucket, contains_genre_type, create_genre_column
from make_plots import (make_frequency_plot, make_line_plot, make_dual_plot_same,
                        make_dual_plot_mixed, make_scatter)
import modeling_functions as mf


# Pipeline

# Import csvs and remove null rows and unnecessary columns
features = clean_features()
weeks = clean_weeks()


# Join tables
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


# Frequency of genres
fig, ax = plt.subplots(figsize=(12, 6))
make_frequency_plot(genresJoined.sort_values(by="SongID", ascending=False), 30, ax)
fig.tight_layout()
fig.suptitle("Frequency of Genres of Tracks", fontsize=20)
fig.subplots_adjust(top=0.9)
#fig.savefig("images/genresJoined.png")


# Frequency of genres (unique)
fig, ax = plt.subplots(figsize=(12, 6))
make_frequency_plot(genresJoined.sort_values(by="SongID", ascending=False), 30, ax)
fig.tight_layout()
fig.suptitle("Frequency of Genres of Tracks (Unique)", fontsize=20)
fig.subplots_adjust(top=0.9)
#fig.savefig("images/genres.png")


# Frequency of genres by decade
decades = ["1960s", "1970s", "1980s", "1990s", "2000s","2010s"]
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
for i, ax in enumerate(axs.flat):
    temp = genresJoinedDecade.loc[genresJoinedDecade['Decade'] == decades[i]] \
            .sort_values(by=['SongID'], ascending=False)
    ax.bar(np.arange(15), temp['SongID'].iloc[0:15])
    ax.set_ylim((0, 24000))
    ax.set_xticks(np.arange(15))
    ax.set_xticklabels(temp['spotify_genre'][0:15], fontsize="large", rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_title(decades[i], fontsize="x-large")
fig.tight_layout()
fig.suptitle("Frequency of Genres of Tracks by Decade", fontsize=28)
fig.subplots_adjust(top=0.9)
#fig.savefig("images/genresJoinedDecade.png")


# Mean of each numerical metric by year
for metric in numericalMetrics:
    fig, ax = plt.subplots()
    make_line_plot(numericals, metric, ax)
    fig.suptitle("Mean {} of Tracks by Year".format(metric.capitalize()), fontsize=20)
    #fig.savefig("images/{}.png".format(metric))


# Test all pairs of columns for correlation coefficient R^2 and select most relevant ones
correlations = list(itertools.combinations(features.columns.tolist()[7:16], 2))
for pair in correlations:
    r2 = stats.pearsonr(features[pair[0]], features[pair[1]])[0]
    if abs(r2) > 0.15:
        pass
        #print("R^2 of " + pair[0] + " and " + pair[1] + " is " + str(r2))


# Dual plots with same y-axis
dualPlotsNormal = [("acousticness", "energy"), ("energy", "danceability"), ("energy", "valence"),
                   ("danceability", "valence")]

for pair in dualPlotsNormal:
    fig, ax = plt.subplots()
    make_dual_plot_same(numericals, pair, ax)
    ax.legend([pair[0].capitalize(), pair[1].capitalize()])
    fig.suptitle("{} and {} of Tracks by Year".format(pair[0].capitalize(),
                 pair[1].capitalize()), fontsize=18)
    #fig.savefig("images/{}and{}.png".format(pair[0], pair[1]))


# Dual plots with mixed y-axes:
dualPlotsMixed = [("energy", "loudness"), ("acousticness", "loudness"), ("energy", "tempo")]
legendLocations = [(0.35, 0.87), (0.55, 0.87), (0.32, 0.87)]

for i, pair in enumerate(dualPlotsMixed):
    fig, ax = plt.subplots()
    make_dual_plot_mixed(numericals, pair, ax)
    fig.legend([pair[0].capitalize(), pair[1].capitalize()], bbox_to_anchor=legendLocations[i])
    fig.suptitle("{} and {} of Tracks by Year".format(pair[0].capitalize(),
                 pair[1].capitalize()), fontsize=18)
    #fig.savefig("images/{}and{}.png".format(pair[0], pair[1]))


# Scatterplots
scatterplots = dualPlotsNormal + dualPlotsMixed
for pair in scatterplots:
    fig, ax = plt.subplots()
    make_scatter(features, pair, ax)
    fig.suptitle("{} vs {} of Tracks".format(pair[0].capitalize(), pair[1].capitalize()),
                 fontsize=20)
    #fig.savefig("images/{}vs{}Scatter.png".format(pair[0], pair[1]))


Xcluster = genreFeatures.set_index('spotify_genre')

Ygroups = []
genreGroupCounts = []
wcss = []
silhouettes = []
for k in range(2, 41):
    km = KMeans(k, n_init=100)
    Ygroup = km.fit_predict(Xcluster)
    counts = Counter(Ygroup)
    Ygroups.append(Ygroup)
    genreGroupCounts.append(counts)
    wcss.append(km.inertia_)
    silhouettes.append(silhouette_score(Xcluster, Ygroup))


# Visualize with elbow method
fig, ax = plt.subplots()
ax.plot(np.arange(2, 41), wcss)
ax.set_xlabel("Number of clusters")
ax.set_ylabel("WCSS")
fig.suptitle("Number of Clusters vs. WCSS")
#fig.savefig("images/elbow.png")

fig, ax = plt.subplots()
ax.plot(np.arange(2, 40), np.abs(np.diff(wcss)), color="C1")
ax2 = ax.twinx()
ax2.plot(np.arange(2, 40), np.abs(np.diff(silhouettes)))
ax.set_xlabel("Number of clusters")
ax.set_ylabel("WCSS")
ax2.set_ylabel("Silhouette score")
fig.legend(["WCSS", "Silhouette"], bbox_to_anchor=(0.8, 0.8))
fig.suptitle("WCSS and Silhouette Score Difference")
#fig.savefig("images/wcssandsilhouettes.png")
# Ideally use 17 clusters


# Dual clustering model: k = 2
km = KMeans(2, n_init=100)
labels = km.fit_predict(Xcluster)
genreBuckets = {i: [] for i in range(2)}
for j in range(len(labels)):
    bucket = labels[j]
    genreBuckets[bucket].append(genres['spotify_genre'][j])


# Initial model with 2 simple buckets
featureBuckets = featureGenres.copy()
featureBuckets['genre_bucket'] = [get_bucket(g, genreBuckets) for g in featureBuckets['spotify_genre']]
featureBuckets = featureBuckets.groupby(['SongID']).mean()
featureBuckets['genre_bucket'] = (featureBuckets['genre_bucket']+0.1).round()

X = featureBuckets[featureBuckets.columns.difference(['genre_bucket'])]
y = featureBuckets['genre_bucket']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Logistic regression model
y_pred, logistic_regression_results = mf.get_logistic_regression_results(X_train, X_test, y_train, y_test)
print(logistic_regression_results)
# 0.8490, 0.9425, 0.8541


# Random forest model
mf.plot_random_forest_hyperparameters(X_train, X_test, y_train, y_test, "binary buckets")

y_pred, random_forest_results = mf.get_random_forest_results(150, 10, 8, X_train, X_test, y_train, y_test)
print(random_forest_results)
# 0.8653, 0.9481, 0.8690


# Gradient boosting model
mf.plot_gradient_boosting_hyperparameters(X_train, X_test, y_train, y_test, "binary buckets")

y_pred, gradient_boosting_results = mf.get_gradient_boosting_results(0.2, 150, 1.0, 9, X_train, X_test, y_train, y_test)
print(gradient_boosting_results)
# 0.8617, 0.9364, 0.8726


# Run models for rock genre
rockColumn = create_genre_column(features['spotify_genre'], "rock")
featureRock = features.assign(is_genre=rockColumn)
featureRock.drop(["Performer", "Song", "spotify_genre"], axis=1, inplace=True)
featureRock.set_index("SongID", inplace=True)
X = featureRock[featureRock.columns.difference(['is_genre'])]
y = featureRock['is_genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Logistic regression model
y_pred, logistic_regression_results = mf.get_logistic_regression_results(X_train, X_test, y_train, y_test)
print(logistic_regression_results)
# 0.8002, 0.9894, 0.8037


# Random forest model
mf.plot_random_forest_hyperparameters(X_train, X_test, y_train, y_test, "rock")

y_pred, random_forest_results = mf.get_random_forest_results(150, 10, 8, X_train, X_test, y_train, y_test)
print(random_forest_results)
# 0.8097, 0.9596, 0.8274


# Gradient boosting model
mf.plot_gradient_boosting_hyperparameters(X_train, X_test, y_train, y_test, "binary buckets")

y_pred, gradient_boosting_results = mf.get_gradient_boosting_results(0.2, 150, 1.0, 9, X_train, X_test, y_train, y_test)
print(gradient_boosting_results)
# 0.8084, 0.9570, 0.8279


# Run models for pop genre
popColumn = create_genre_column(features['spotify_genre'], "pop")
featurePop = features.assign(is_genre=popColumn)
featurePop.drop(["Performer", "Song", "spotify_genre"], axis=1, inplace=True)
featurePop.set_index("SongID", inplace=True)
X = featurePop[featurePop.columns.difference(['is_genre'])]
y = featurePop['is_genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Logistic regression model
y_pred, logistic_regression_results = mf.get_logistic_regression_results(X_train, X_test, y_train, y_test)
print(logistic_regression_results)
# 0.8034, 0.9960, 0.8056


# Random forest model
mf.plot_random_forest_hyperparameters(X_train, X_test, y_train, y_test, "pop")

y_pred, random_forest_results = mf.get_random_forest_results(150, 10, 8, X_train, X_test, y_train, y_test)
print(random_forest_results)
# 0.8183, 0.9786, 0.8273


# Gradient boosting model
mf.plot_gradient_boosting_hyperparameters(X_train, X_test, y_train, y_test, "binary buckets")

y_pred, gradient_boosting_results = mf.get_gradient_boosting_results(0.2, 150, 1.0, 9, X_train, X_test, y_train, y_test)
print(gradient_boosting_results)
# 0.8174, 0.9786, 0.8265
