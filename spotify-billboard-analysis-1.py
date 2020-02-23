import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
from clean_features import clean_features
from clean_weeks import clean_weeks
from make_plots import *


# Import csvs and remove null rows and unnecessary columns
features = clean_features()
weeks = clean_weeks()


# Derived dataframes
featuresNoNulls = features.dropna()

joined = weeks.merge(features, on='SongID')
#joined.to_csv("data/joined.csv", index=False)

joinedNoNulls = weeks.merge(featuresNoNulls, on='SongID')

featureGenres = features.explode('spotify_genre')
featureGenres = featureGenres[featureGenres.spotify_genre != '']

joinedGenres = joined.explode('spotify_genre')
joinedGenres = joinedGenres[joinedGenres.spotify_genre != '']

# Create grouped tables
genres = featureGenres.groupby(['spotify_genre'])['SongID'].count().reset_index()
genresJoined = joinedGenres.groupby(['spotify_genre'])['SongID'].count().reset_index()
genresJoinedDecade = joinedGenres.groupby(['spotify_genre", "Decade'])['SongID'].count(). \
                        reset_index().sort_values(by="Decade")


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
explicitness = joined[['Year', 'spotify_track_explicit']].dropna()
explicitness = explicitness.groupby(['Year']).mean().reset_index()
fig, ax = plt.subplots()
ax.plot(explicitness['Year'], explicitness['num'])
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("Proportion of explicit songs", fontsize=14)
fig.suptitle("Explicitness of Tracks", fontsize=20)
fig.savefig("images/explicitness.png")


# Mean of each numerical metric by year
numericals = ['Year'] + joined.columns.tolist()[11:22]
numericalMetrics = joined[numericals].groupby(['Year']).aggregate(np.nanmean).reset_index()


for metric in numericalMetrics:
    fig, ax = plt.subplots()
    make_line_plot(numericals, metric, ax)
    fig.suptitle("Mean {} of Tracks by Year".format(metric.capitalize()), fontsize=20)
    #fig.savefig("images/{}.png".format(metric))


# Test all pairs of columns for correlation coefficient R^2 and select most relevant ones
correlations = list(itertools.combinations(features.columns.tolist()[6:16], 2))
for pair in correlations:
    r2 = stats.pearsonr(featuresNoNulls[pair[0]], featuresNoNulls[pair[1]])[0]
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


# Hypothesis test
hypothesisTests = ['energy", "danceability", "loudness", "valence", "tempo']
hypothesisTestMetrics = joinedNoNulls[['Decade'] + hypothesisTests]
# Ho = Music today has the same energy, danceability, loudness, valence, and
#       tempo as music of 60 years ago.
# Ha = Music today does not have the same energy, danceability, loudness, valence, and
#       tempo as music of 60 years ago.
# alpha = 0.05
print("\n")

for metric in hypothesisTests:
    l1 = hypothesisTestMetrics[metric].loc[hypothesisTestMetrics['Decade'] == "1960s"]
    l2 = hypothesisTestMetrics[metric].loc[hypothesisTestMetrics['Decade'] == "2010s"]
    t, p = stats.ttest_ind(l1, l2, nan_policy="omit")
    print("{}: u = {}, p = {}".format(metric, t, p))
print("\n")

for metric in hypothesisTests:
    l1 = hypothesisTestMetrics[metric].loc[hypothesisTestMetrics['Decade'] == "1960s"]
    l2 = hypothesisTestMetrics[metric].loc[hypothesisTestMetrics['Decade'] == "2010s"]
    u, p = stats.mannwhitneyu(l1, l2, alternative="two-sided")
    print("{}: u = {}, p = {}".format(metric, u, p))
