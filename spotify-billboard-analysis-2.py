import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
from collections import Counter
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import log_loss, make_scorer, silhouette_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from statsmodels.tsa.arima_model import ARIMA


# Import csvs and remove null rows and unnecessary columns
weeks = pd.read_csv("data/hot-stuff.csv", converters={'WeekID': lambda d: pd.to_datetime(d, \
                    format="%m/%d/%Y", errors="coerce")})
weeksFilter = ['url', 'Instance', 'Previous Week Position', 'Peak Position',
               'Weeks on Chart']
weeks.drop(weeksFilter, axis=1, inplace=True)

features = pd.read_csv("data/hot-100-audio-features.csv", converters={'spotify_genre':
                       lambda s: s[1:-1].split(', ')}, encoding="latin-1")
features['spotify_genre'] = features['spotify_genre'].apply(lambda l: [s[1:-1] for s in l])
featuresFilter = ["spotify_track_id", "spotify_track_preview_url", "spotify_track_album",
                  "spotify_track_popularity", "key", "time_signature"]
features.drop(featuresFilter, axis=1, inplace=True)
features.dropna(inplace=True)


# Column insertion
def decade(year: int) -> str:
    if year >= 1950 and year < 1960:
        return "1950s"
    elif year >= 1960 and year < 1970:
        return "1960s"
    elif year >= 1970 and year < 1980:
        return "1970s"
    elif year >= 1980 and year < 1990:
        return "1980s"
    elif year >= 1990 and year < 2000:
        return "1990s"
    elif year >= 2000 and year < 2010:
        return "2000s"
    elif year >= 2010 and year < 2020:
        return "2010s"
    else:
        return None

weeks.insert(1, "Month", weeks['WeekID'].dt.month)
weeks.insert(2, "Year", weeks['WeekID'].dt.year)
weeks.insert(3, "Decade", weeks['Year'].apply(decade))

features['spotify_track_explicit'] = features['spotify_track_explicit'].astype(float)
features.insert(6, "track_duration", features['spotify_track_duration_ms'] / 1000)

# Join tables
joined = weeks.merge(features, on='SongID')
joined.to_csv("data/joined.csv", index=False)

# Expand genres into individual components
featureGenres = features.explode('spotify_genre')
featureGenres = featureGenres[featureGenres.spotify_genre != '']
featureGenres = featureGenres.dropna()

joinedGenres = joined.explode('spotify_genre')
joinedGenres = joinedGenres[joinedGenres.spotify_genre != '']
joinedGenres = joinedGenres.dropna()

explicitness = joined[['Year', 'spotify_track_explicit']].dropna()
explicitness = explicitness.groupby(['Year']).aggregate(np.nanmean).reset_index()

numericalMetrics = [joined.columns.tolist()[11]] + joined.columns.tolist()[13:24]
numericals = joined[['Year'] + numericalMetrics].groupby(['Year']).aggregate(np.nanmean).reset_index()


# Create grouped tables
genresJoined = joinedGenres.groupby(['spotify_genre'])['SongID'].count().reset_index(). \
                   sort_values(by=['SongID'], ascending=False)
genres = featureGenres.groupby(['spotify_genre'])['SongID'].count().reset_index(). \
            sort_values(by=['SongID'], ascending=False)
genresJoinedDecade = joinedGenres.groupby(['spotify_genre', 'Decade'])['SongID'].count(). \
                        reset_index().sort_values(by=['SongID'], ascending=False)


# Only keep top 200 recorded genres
temp = np.diff(genres['SongID'])
topGenres = list(genres['spotify_genre'].iloc[0:200])
featuresCulled = featureGenres.loc[featureGenres['spotify_genre'].isin(topGenres), :]
joinedCulled = joinedGenres.loc[joinedGenres['spotify_genre'].isin(topGenres), :]

genreFeatures = featureGenres.groupby(['spotify_genre'])[numericalMetrics].mean().reset_index()
genreFeaturesCulled = featuresCulled.groupby(['spotify_genre'])[numericalMetrics].mean().reset_index()

# print(len(pd.unique(features['SongID'])))
# print(len(pd.unique(featuresCulled['SongID'])))


def make_frequency_plot(df: pd.DataFrame, top: int, ax: plt.axes) -> None:
    ax.bar(np.arange(top), genresJoined['SongID'].iloc[0:top])
    ax.set_xticks(np.arange(top))
    ax.set_xticklabels(df['spotify_genre'][0:top], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Genre", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)

# Frequency of genres
fig, ax = plt.subplots(figsize=(12, 6))
make_frequency_plot(genresJoined, 30, ax)
fig.tight_layout()
fig.suptitle("Frequency of Genres of Tracks", fontsize=20)
fig.subplots_adjust(top=0.9)
fig.savefig("images/genresJoined.png")


# Frequency of genres (unique)
fig, ax = plt.subplots(figsize=(12, 6))
make_frequency_plot(genres, 30, ax)
fig.tight_layout()
fig.suptitle("Frequency of Genres of Tracks (Unique)", fontsize=20)
fig.subplots_adjust(top=0.9)
fig.savefig("images/genres.png")


# Frequency of genres by decade
decades = ["1960s", "1970s", "1980s", "1990s", "2000s","2010s"]
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
for i, ax in enumerate(axs.flat):
    temp = genresJoinedDecade.loc[genresJoinedDecade['Decade'] == decades[i]]
    ax.bar(np.arange(15), temp['SongID'].iloc[0:15])
    ax.set_ylim((0, 24000))
    ax.set_xticks(np.arange(15))
    ax.set_xticklabels(temp['spotify_genre'][0:15], fontsize="large", rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_title(decades[i], fontsize="x-large")
fig.tight_layout()
fig.suptitle("Frequency of Genres of Tracks by Decade", fontsize=28)
fig.subplots_adjust(top=0.9)
fig.savefig("images/genresJoinedDecade.png")


# Explicitness
fig, ax = plt.subplots()
ax.plot(explicitness['Year'], explicitness['spotify_track_explicit'])
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("Proportion of explicit songs", fontsize=14)
fig.suptitle("Explicitness of Tracks", fontsize=20)
fig.savefig("images/explicitness.png")


# Mean of each numerical metric by year
def make_line_plot(df: pd.DataFrame, col: str, ax: plt.axes) -> None:
    ax.plot(df['Year'], df[col])
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("{}".format(col.capitalize()), fontsize=14)

for metric in numericalMetrics:
    fig, ax = plt.subplots()
    make_line_plot(numericals, metric, ax)
    fig.suptitle("Mean {} of Tracks by Year".format(metric.capitalize()), fontsize=20)
    fig.savefig("images/{}.png".format(metric))


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

def make_dual_plot_same(df: pd.DataFrame, pair: tuple, ax: plt.axes) -> None:
    ax.plot(df['Year'], df[pair[0]])
    ax.plot(df['Year'], df[pair[1]])
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("{}".format("Value"), fontsize=12)

for pair in dualPlotsNormal:
    fig, ax = plt.subplots()
    make_dual_plot_same(numericals, pair, ax)
    ax.legend([pair[0].capitalize(), pair[1].capitalize()])
    fig.suptitle("{} and {} of Tracks by Year".format(pair[0].capitalize(),
                 pair[1].capitalize()), fontsize=18)
    fig.savefig("images/{}and{}.png".format(pair[0], pair[1]))


# Dual plots with mixed y-axes:
dualPlotsMixed = [("energy", "loudness"), ("acousticness", "loudness"), ("energy", "tempo")]
legendLocations = [(0.35, 0.87), (0.55, 0.87), (0.32, 0.87)]

def make_dual_plot_mixed(df: pd.DataFrame, pair: tuple, ax: plt.axes) -> None:
    ax.plot(numericals['Year'], numericals[pair[0]])
    ax2 = ax.twinx()
    ax2.plot(numericals['Year'], numericals[pair[1]], color="C1")
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel(pair[0].capitalize(), fontsize=14)
    if pair[1] == 'loudness':
        ax2.set_ylabel("Loudness (dB)", fontsize=14)
    elif pair[1] == 'tempo':
        ax2.set_ylabel("Tempo (bpm)", fontsize=14)
    else:
        ax2.set_ylabel("{}".format("Value"), fontsize=14)

for i, pair in enumerate(dualPlotsMixed):
    fig, ax = plt.subplots()
    make_dual_plot_mixed(numericals, pair, ax)
    fig.legend([pair[0].capitalize(), pair[1].capitalize()], bbox_to_anchor=legendLocations[i])
    fig.suptitle("{} and {} of Tracks by Year".format(pair[0].capitalize(),
                 pair[1].capitalize()), fontsize=18)
    fig.savefig("images/{}and{}.png".format(pair[0], pair[1]))


# Scatterplots
def make_scatter(df: pd.DataFrame, pair: tuple, ax: plt.axes) -> None:
    ax.scatter(df[pair[0]], df[pair[1]])
    ax.set_xlabel(pair[0].capitalize(), fontsize=14)
    ax.set_ylabel(pair[1].capitalize(), fontsize=14)
    r2 = stats.pearsonr(df[pair[0]], df[pair[1]])[0]
    print("\nR^2 of " + pair[0] + " and " + pair[1] + " is " + str(r2))

scatterplots = dualPlotsNormal + dualPlotsMixed
for pair in scatterplots:
    fig, ax = plt.subplots()
    make_scatter(features, pair, ax)
    fig.suptitle("{} vs {} of Tracks".format(pair[0].capitalize(), pair[1].capitalize()),
                 fontsize=20)

    fig.savefig("images/{}vs{}Scatter.png".format(pair[0], pair[1]))


# Find genres that are closest to each other using table of means
Xcluster = genreFeatures.set_index('spotify_genre')
genreGroupCounts = []
wcss = []
for k in range(5, 41):
    km = KMeans(k)
    Ygroup = km.fit_predict(Xcluster)
    counts = Counter(Ygroup)
    genreGroupCounts.append(dict(counts))
    wcss.append(km.inertia_)

print(genreGroupCounts)
print(wcss)


def manhattan_distance(l1: list, l2: list) -> float:
    if len(l1) != len(l2):
        return -1
    dist = 0
    for i in range(len(l1)):
        dist += abs(l1[i], l2[i])
    return dist

