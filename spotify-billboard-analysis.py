import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
#import spotipy

# Import csvs and remove null rows and unnecessary columns
weeks = pd.read_csv("data/hot-stuff.csv", converters={'WeekID': lambda d: pd.to_datetime(d, \
                    format="%m/%d/%Y", errors="coerce")})
weeksFilter = ['url', 'Instance', 'Previous Week Position', 'Peak Position',
               'Weeks on Chart']
weeks.drop(weeksFilter, axis=1, inplace=True)

features = pd.read_csv("data/hot-100-audio-features.csv", converters={'spotify_genre':
                       lambda s: s[1:-1].split(', ')}, encoding="latin-1")
features['spotify_genre'] = features['spotify_genre'].apply(lambda l: [s[1:-1] for s in l])
features.dropna(subset=['spotify_genre'])
featuresFilter = ['spotify_track_id', 'spotify_track_preview_url', 'spotify_track_album',
                  'spotify_track_popularity', 'key', 'mode', 'time_signature']
features.drop(featuresFilter, axis=1, inplace=True)

# Derived dataframes

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

weeks.insert(1, 'Year', weeks['WeekID'].dt.year)
weeks.insert(2, 'Decade', weeks['Year'].apply(decade))

features.insert(7, 'track_duration', features['spotify_track_duration_ms'] / 1000)
featuresNoNulls = features.dropna()

joined = weeks.merge(features, on='SongID')
joined.to_csv("data/joined.csv", index=False)

joinedNoNulls = weeks.merge(featuresNoNulls, on='SongID')

featureGenres = features.explode('spotify_genre')
featureGenres = featureGenres[featureGenres.spotify_genre != '']

joinedGenres = joined.explode('spotify_genre')
joinedGenres = joinedGenres[joinedGenres.spotify_genre != '']


# Frequency of genres
genresJoined = joinedGenres.groupby(['spotify_genre'])['SongID'].count().reset_index(). \
                   sort_values(by=['SongID'], ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(np.arange(30), genresJoined['SongID'].iloc[0:30])
ax.set_xticks(np.arange(30))
ax.set_xticklabels(genresJoined['spotify_genre'][0:30], rotation=45, ha="right",
                   rotation_mode="anchor")
ax.set_xlabel("Genre")
fig.tight_layout()
fig.suptitle("Frequency of Genres of Billboard Songs", fontsize=20)
fig.subplots_adjust(top=0.9)
fig.savefig("images/genresJoined.png")


# Frequency of genres (unique)
genres = featureGenres.groupby(['spotify_genre'])['SongID'].count().reset_index(). \
             sort_values(by=['SongID'], ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(np.arange(30), genres['SongID'].iloc[0:30])
ax.set_xticks(np.arange(30))
ax.set_xticklabels(genres['spotify_genre'][0:30], rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel("Genre")
fig.tight_layout()
fig.suptitle("Frequency of Genres of Billboard Songs (Unique)", fontsize=20)
fig.subplots_adjust(top=0.9)
fig.savefig("images/genres.png")



# Frequency of genres by decade
genresJoinedDecade = joinedGenres.groupby(['spotify_genre', 'Decade'])['SongID'].count(). \
                         reset_index().sort_values(by=['SongID'], ascending=False)
decades = ["1960s", "1970s", "1980s", "1990s", "2000s","2010s"]
fig, axs = plt.subplots(3, 2, figsize=(14, 14))
for i, ax in enumerate(axs.flat):
    temp = genresJoinedDecade.loc[genresJoinedDecade['Decade'] == decades[i]]
    ax.bar(np.arange(15), temp['SongID'].iloc[0:15])
    ax.set_ylim((0, 24000))
    ax.set_xticks(np.arange(15))
    ax.set_xticklabels(temp['spotify_genre'][0:15], fontsize="large", rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_title(decades[i], fontsize="large")
fig.tight_layout()
fig.suptitle("Frequency of Genres of Billboard Songs by Decade", fontsize=28)
fig.subplots_adjust(top=0.9)
fig.savefig("images/genresJoinedDecade.png")



'''
# Frequency of genres by decade
genresJoinedDecade = joinedGenres.groupby(['spotify_genre', 'Decade'])['SongID'].count(). \
                        reset_index().sort_values(by=['SongID'], ascending=False)
decades = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]

def stacked_bar_helper(axis, ) -> None:
    pass

genreList = {}
for d in decades:
    temp = genresJoinedDecade.loc[genresJoinedDecade['Decade'] == d]
    genreList[d] = temp['spotify_genre'].iloc[0:15]
fig, ax = plt.subplots()
for d in genreList.keys():
    temp = genresJoinedDecade.loc[genresJoinedDecade['Decade'] == d]
    ax.bar(np.arange(len(decades)), temp['SongID'].iloc[0:15])
ax.set_xticks(np.arange(len(decades)))
ax.set_xticklabels(decades, fontsize="large", rotation=45, ha="right", rotation_mode="anchor")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper center')
fig.suptitle("Frequency of Genres of Billboard Songs by Decade", fontsize=28)
fig.subplots_adjust(top=0.9)
fig.savefig("images/genresJoinedDecade.png")
'''


# Explicitness
explicitness = joined[['Year', 'spotify_track_explicit']].dropna()
explicitness['num'] = explicitness['spotify_track_explicit'].astype(int)
explicitness = explicitness.groupby(['Year']).aggregate(np.nanmean).reset_index()
fig, ax = plt.subplots()
ax.plot(explicitness['Year'], explicitness['num'])
ax.set_xlabel("Year")
fig.suptitle("Explicitness of Billboard Songs", fontsize=14)
fig.savefig("images/explicitness.png")


# Mean of each numerical metric by year
numericals = ['Year'] + joined.columns.tolist()[12:22]
numericalMetrics = joined[numericals].groupby(['Year']).aggregate(np.nanmean).reset_index()
for metric in numericalMetrics.columns.tolist()[1:]:
    fig, ax = plt.subplots()
    ax.plot(numericalMetrics['Year'], numericalMetrics[metric])
    ax.set_xlabel("Year")
    ax.set_ylabel("{}".format(metric.capitalize()))
    fig.suptitle("Mean {} of Billboard Songs by Year".format(metric.capitalize()), fontsize=14)
    fig.savefig("images/{}.png".format(metric))


# Test all pairs of columns for correlation coefficient R^2 and select most relevant ones
correlations = list(itertools.combinations(features.columns.tolist()[7:16], 2))
for pair in correlations:
    r2 = stats.pearsonr(featuresNoNulls[pair[0]], featuresNoNulls[pair[1]])[0]
    if abs(r2) > 0.15:
        print("\nR^2 of " + pair[0] + " and " + pair[1] + " is " + str(r2))


# Dual plots with same y-axis
dualPlotsNormal = [('acousticness', 'energy'), ('energy', 'danceability'), ('energy', 'valence'),
                   ('danceability', 'valence')]
for pair in dualPlotsNormal:
    fig, ax = plt.subplots()
    ax.plot(numericalMetrics['Year'], numericalMetrics[pair[0]])
    ax.plot(numericalMetrics['Year'], numericalMetrics[pair[1]])
    ax.set_xlabel("Year")
    ax.set_ylabel("{}".format("Value"))
    ax.legend([pair[0].capitalize(), pair[1].capitalize()])
    fig.suptitle("Mean {} and {} of Billboard Songs by Year".format(pair[0].capitalize(),
                 pair[1].capitalize()), fontsize=12)
    fig.savefig("images/{}and{}.png".format(pair[0], pair[1]))


# Dual plots with mixed y-axes:
dualPlotsMixed = [('energy', 'loudness'), ('acousticness', 'loudness'), ('energy', 'tempo')]
legendLocations = [(0.35, 0.87), (0.55, 0.87), (0.32, 0.87)]
for i, pair in enumerate(dualPlotsMixed):
    fig, ax = plt.subplots()
    l1 = ax.plot(numericalMetrics['Year'], numericalMetrics[pair[0]])
    ax2 = ax.twinx()
    l2 = ax2.plot(numericalMetrics['Year'], numericalMetrics[pair[1]], color="C1")
    ax.set_xlabel("Year")
    ax.set_ylabel(pair[0])
    ax2.set_ylabel(pair[1])
    fig.legend([pair[0].capitalize(), pair[1].capitalize()], bbox_to_anchor=legendLocations[i])
    fig.suptitle("Mean {} and {} of Billboard Songs by Year".format(pair[0].capitalize(),
                 pair[1].capitalize()), fontsize=12)
    fig.savefig("images/{}and{}.png".format(pair[0], pair[1]))


# Scatterplots
scatterplots = dualPlotsNormal + dualPlotsMixed
for pair in scatterplots:
    fig, ax = plt.subplots()
    ax.scatter(featuresNoNulls[pair[0]], featuresNoNulls[pair[1]])
    ax.set_xlabel(pair[0].capitalize())
    ax.set_ylabel(pair[1].capitalize())
    fig.suptitle("{} vs {} of Billboard Songs".format(pair[0].capitalize(), pair[1].capitalize()),
                 fontsize=14)
    fig.savefig("images/{}vs{}Scatter.png".format(pair[0], pair[1]))
    r2 = stats.pearsonr(featuresNoNulls[pair[0]], featuresNoNulls[pair[1]])[0]
    print("\nR^2 of " + pair[0] + " and " + pair[1] + " is " + str(r2))


# Hypothesis test
hypothesisTests = ['energy', 'danceability', 'loudness', 'valence', 'tempo']
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
