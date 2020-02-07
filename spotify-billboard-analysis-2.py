import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.model_selection import train_test_split, GridSearchCV


# Pipeline

# Import csvs and remove null rows and unnecessary columns
weeks = pd.read_csv("data/hot-stuff.csv", converters={'WeekID': lambda d: pd.to_datetime(d, \
                    format="%m/%d/%Y", errors="coerce")})
weeksFilter = ["url", "Instance", "Previous Week Position", "Peak Position",
               "Weeks on Chart"]
weeks.drop(weeksFilter, axis=1, inplace=True)

features = pd.read_csv("data/hot-100-audio-features.csv", converters={'spotify_genre':
                       lambda s: s[1:-1].split(", ")}, encoding="latin-1")
featuresFilter = ["spotify_track_id", "spotify_track_preview_url", "spotify_track_album",
                  "spotify_track_popularity", "key", "time_signature"]
features.drop(featuresFilter, axis=1, inplace=True)
emptyGenreRows = []
for row in range(len(features['spotify_genre'])):
    if features['spotify_genre'][row] == ['']:
        emptyGenreRows.append(row)
features.drop(emptyGenreRows, axis=0, inplace=True)
features = features[features['tempo'] != 0]
features.dropna(inplace=True)
features['spotify_genre'] = features['spotify_genre'].map(lambda l: [s[1:-1] for s in l])


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

weeks.insert(1, "Year", weeks['WeekID'].dt.year)
weeks.insert(2, "Decade", weeks['Year'].apply(decade))

features['spotify_track_explicit'] = features['spotify_track_explicit'].astype(float)
features['spotify_track_duration_ms'] = features['spotify_track_duration_ms'] / 1000
features.rename(columns={"spotify_track_duration_ms": "track_duration"}, inplace=True)


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


def make_frequency_plot(df: pd.DataFrame, top: int, ax: plt.axes) -> None:
    ax.bar(np.arange(top), df['SongID'].iloc[0:top])
    ax.set_xticks(np.arange(top))
    ax.set_xticklabels(df['spotify_genre'][0:top], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Genre", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)

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


# Explicitness
fig, ax = plt.subplots()
ax.plot(explicitness['Year'], explicitness['spotify_track_explicit'])
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("Proportion of explicit songs", fontsize=14)
fig.suptitle("Explicitness of Tracks", fontsize=20)
#fig.savefig("images/explicitness.png")


# Mean of each numerical metric by year
def make_line_plot(df: pd.DataFrame, col: str, ax: plt.axes) -> None:
    ax.plot(df['Year'], df[col])
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("{}".format(col.capitalize()), fontsize=14)

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
    #fig.savefig("images/{}and{}.png".format(pair[0], pair[1]))


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
    #fig.savefig("images/{}and{}.png".format(pair[0], pair[1]))



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
    #fig.savefig("images/{}vs{}Scatter.png".format(pair[0], pair[1]))


Xcluster = genreFeatures.set_index('spotify_genre')
'''
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
fig.savefig("images/elbow.png")

fig, ax = plt.subplots()
ax.plot(np.arange(41-2), np.abs(np.diff(wcss)), color="C1")
ax2 = ax.twinx()
ax2.plot(np.arange(41-2), np.abs(np.diff(silhouettes)))
fig.savefig("images/wcssandsilhouettes.png")
'''


# Dual clustering model: k = 2
km = KMeans(2, n_init=100)
labels = km.fit_predict(Xcluster)
genreBuckets = {i: [] for i in range(2)}
for j in range(len(labels)):
    bucket = labels[j]
    genreBuckets[bucket].append(genres['spotify_genre'][j])


# Add column to data frame
def get_bucket(genre: str, buckets: dict) -> int:
    for key, value in buckets.items():
        if genre in value:
            return key
    return float("nan")


# Keep only top 200 genres if applicable (eliminates 80% of genres for only 10% loss of data)
topGenres = list(genres.sort_values(by="SongID", ascending=False)['spotify_genre'][0:100])
featuresTopGenres = featureGenres[featureGenres['spotify_genre'].isin(topGenres)]


def create_confusion_matrix(ytest: np.array, ypred: np.array) -> (int, int, int, int):
    cm = confusion_matrix(ytest, ypred)
    tp = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[1, 1]
    return tp, fp, fn, tn

def get_precision_recall(tp: int, fp: int, fn: int, tn: int) -> float:
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


# Logistic Regression wrapper function
def get_logistic_regression_results(xtrain: np.array, xtest: np.array, ytrain: np.array, \
                                    ytest: np.array) -> (float, float, float):
    lr = LogisticRegression(C=1000, max_iter=1000).fit(xtrain, ytrain)
    y_pred = lr.predict(xtest)
    accuracy = lr.score(xtest, ytest)
    tp, fp, fn, tn = create_confusion_matrix(ytest, y_pred)
    precision, recall = get_precision_recall(tp, fp, fn, tn)
    return accuracy, precision, recall


def plot_random_forest_hyperparameters(xtrain: np.array, xtest: np.array, ytrain: np.array, \
                                       ytest: np.array, genre_type: str) -> None:
    # Find optimal number of trees
    numTrees = np.arange(50, 201, 30)
    accuracy_t = []
    for n in numTrees:
        a = 0
        for i in range(5):
            rf = RandomForestClassifier(n, oob_score=True, n_jobs=-1).fit(xtrain, ytrain)
            # y_pred = rf.predict(xtest)
            a += rf.score(xtest, ytest) / 5
        accuracy_t.append(a)
    fig, ax = plt.subplots()
    ax.plot(numTrees, accuracy_t)
    ax.set_title("RF accuracy by number of trees ({})".format(genre_type))

    # Find optimal max depth
    maxDepth = np.arange(3, 13)
    accuracy_d = []
    for d in maxDepth:
        a = 0
        for i in range(5):
            rf = RandomForestClassifier(max_depth=d, oob_score=True, n_jobs=-1). \
                    fit(xtrain, ytrain)
            # y_pred = rf.predict(xtest)
            a += rf.score(xtest, ytest) / 5
        accuracy_d.append(a)
    fig, ax = plt.subplots()
    ax.plot(maxDepth, accuracy_d)
    ax.set_title("RF accuracy by max depth ({})".format(genre_type))

    # Find optimal number of features
    maxFeatures = np.arange(5, 11)
    accuracy_f = []
    for f in maxFeatures:
        a = 0
        for i in range(5):
            rf = RandomForestClassifier(max_features=f, oob_score=True, n_jobs=-1). \
                    fit(xtrain, ytrain)
            # y_pred = rf.predict(xtest)
            a += rf.score(xtest, ytest) / 5
        accuracy_f.append(a)
    fig, ax = plt.subplots()
    ax.plot(maxFeatures, accuracy_f)
    ax.set_title("RF accuracy by max features ({})".format(genre_type))


# Random Forest wrapper function
def get_random_forest_results(num_trees: int, max_depth: int, max_features: int, \
                              xtrain: np.array, xtest: np.array, ytrain: np.array, \
                              ytest: np.array) -> (float, float, float):
    rf = RandomForestClassifier(num_trees, max_features=max_features, oob_score=True,\
                                n_jobs=-1).fit(xtrain, ytrain)
    y_pred = rf.predict(xtest)
    accuracy = rf.score(xtest, ytest)
    tp, fp, fn, tn = create_confusion_matrix(ytest, y_pred)
    precision, recall = get_precision_recall(tp, fp, fn, tn)
    return (accuracy, precision, recall)


def plot_gradient_boosting_hyperparameters(xtrain: np.array, xtest: np.array, ytrain: np.array, \
                                           ytest: np.array, genre_type: str) -> None:
    # Find optimal learning rate
    learningRate = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    accuracy_l = []
    for l in learningRate:
        gbr = GradientBoostingClassifier(learning_rate=l).fit(xtrain, ytrain)
        # y_pred = gbr.predict(xtest)
        a = gbr.score(xtest, ytest)
        accuracy_l.append(a)
    fig, ax = plt.subplots()
    ax.plot(learningRate, accuracy_l)
    ax.set_title("GBR accuracy by learning rate ({})".format(genre_type))

    # Find optimal number of trees
    numTrees = np.arange(50, 201, 30)
    accuracy_t = []
    for n in numTrees:
        gbr = GradientBoostingClassifier(n_estimators=n).fit(xtrain, ytrain)
        # y_pred = gbr.predict(xtest)
        a = gbr.score(xtest, ytest)
        accuracy_t.append(a)
    fig, ax = plt.subplots()
    ax.plot(numTrees, accuracy_t)
    ax.set_title("GBR accuracy by number of trees ({})".format(genre_type))

    # Find optimal subsample rate
    subsampleRate = np.arange(0.2, 1.1, 0.2)
    accuracy_s = []
    for s in subsampleRate:
        gbr = GradientBoostingClassifier(subsample=s).fit(xtrain, ytrain)
        # y_pred = gbr.predict(xtest)
        a = gbr.score(xtest, ytest)
        accuracy_s.append(a)
    fig, ax = plt.subplots()
    ax.plot(subsampleRate, accuracy_s)
    ax.set_title("GBR accuracy by subsample rate ({})".format(genre_type))

    # Find optimal max depth
    maxDepth = np.arange(3, 13)
    accuracy_d = []
    for d in maxDepth:
        gbr = GradientBoostingClassifier(max_depth=d).fit(xtrain, ytrain)
        # y_pred = gbr.predict(xtest)
        a = gbr.score(xtest, ytest)
        accuracy_d.append(a)
    fig, ax = plt.subplots()
    ax.plot(maxDepth, accuracy_d)
    ax.set_title("GBR accuracy by max depth ({})".format(genre_type))


# Gradient boosting wrapper function
def get_gradient_boosting_results(learning_rate: float, num_trees: int, subsample_rate: float, \
                                  max_features: int, xtrain: np.array, xtest: np.array, \
                                  ytrain: np.array, ytest: np.array) -> (float, float, float):
    gbr = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=num_trees, \
                                     subsample=subsample_rate, max_features=max_features).fit(xtrain, ytrain)
    y_pred = gbr.predict(xtest)
    accuracy = gbr.score(xtest, ytest)
    tp, fp, fn, tn = create_confusion_matrix(ytest, y_pred)
    precision, recall = get_precision_recall(tp, fp, fn, tn)
    return (accuracy, precision, recall)


# Initial model with 2 simple buckets
featureBuckets = featureGenres.copy()
featureBuckets['genre_bucket'] = [get_bucket(g, genreBuckets) for g in featureBuckets['spotify_genre']]
featureBuckets = featureBuckets.groupby(['SongID']).mean()
featureBuckets['genre_bucket'] = (featureBuckets['genre_bucket']+0.1).round()

X = featureBuckets[featureBuckets.columns.difference(['genre_bucket'])]
y = featureBuckets['genre_bucket']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Logistic regression model
logistic_regression_results = get_logistic_regression_results(X_train, X_test, y_train, y_test)
print(logistic_regression_results)
# 0.8490, 0.9425, 0.8541


# Random forest model
plot_random_forest_hyperparameters(X_train, X_test, y_train, y_test, "binary buckets")

random_forest_results = get_random_forest_results(150, 10, 8, X_train, X_test, y_train, y_test)
print(random_forest_results)
# 0.8653, 0.9481, 0.8690


# Gradient boosting model
plot_gradient_boosting_hyperparameters(X_train, X_test, y_train, y_test, "binary buckets")

gradient_boosting_results = get_gradient_boosting_results(0.2, 150, 1.0, 9, X_train, X_test, y_train, y_test)
print(gradient_boosting_results)


# Create binary classfier for each genre group (rock, pop, hip hop/rap, jazz, etc)
def genre_type_classifier(genre: str, genre_types: list) -> int:
    for g in genre_types:
        if g in genre:
            return 1
    return 0

def create_df_with_genres(genres: list):
    df = featureGenres.copy()
    df['genre_bucket'] = [genre_type_classifier(g, genres) for g in df['spotify_genre']]
    df = df.groupby(['SongID']).mean()
    df['genre_bucket'] = (df['genre_bucket']+0.1).round()
    return df


# Run models for rock genre
featureRock = create_df_with_genres(["rock"])
X = featureRock[featureRock.columns.difference(['genre_bucket'])]
y = featureRock['genre_bucket']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Logistic regression model
logistic_regression_results = get_logistic_regression_results(X_train, X_test, y_train, y_test)
print(logistic_regression_results)
# 0.8002, 0.9894, 0.8037


# Random forest model
plot_random_forest_hyperparameters(X_train, X_test, y_train, y_test, "rock")

random_forest_results = get_random_forest_results(150, 10, 8, X_train, X_test, y_train, y_test)
print(random_forest_results)
# 0.8097, 0.9596, 0.8274


# Gradient boosting model
plot_gradient_boosting_hyperparameters(X_train, X_test, y_train, y_test, "binary buckets")

gradient_boosting_results = get_gradient_boosting_results(0.2, 150, 1.0, 9, X_train, X_test, y_train, y_test)
print(gradient_boosting_results)
# 0.8084, 0.9570, 0.8279


# Run models for pop genre
featureRock = create_df_with_genres(["pop"])
X = featureRock[featureRock.columns.difference(['genre_bucket'])]
y = featureRock['genre_bucket']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Logistic regression model
logistic_regression_results = get_logistic_regression_results(X_train, X_test, y_train, y_test)
print(logistic_regression_results)
# 0.8034, 0.9960, 0.8056


# Random forest model
plot_random_forest_hyperparameters(X_train, X_test, y_train, y_test, "rock")

random_forest_results = get_random_forest_results(150, 10, 8, X_train, X_test, y_train, y_test)
print(random_forest_results)
# 0.8183, 0.9786, 0.8273


# Gradient boosting model
plot_gradient_boosting_hyperparameters(X_train, X_test, y_train, y_test, "binary buckets")

gradient_boosting_results = get_gradient_boosting_results(0.2, 150, 1.0, 9, X_train, X_test, y_train, y_test)
print(gradient_boosting_results)
#
