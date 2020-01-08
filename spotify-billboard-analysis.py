import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools

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

featuresScatter = features.dropna()

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

joined = weeks.merge(features, on='SongID')
joined.to_csv("data/joined.csv", index=False)

featureGenres = features.explode('spotify_genre')
featureGenres = featureGenres[featureGenres.spotify_genre != '']

joinedGenres = joined.explode('spotify_genre')
joinedGenres = joinedGenres[joinedGenres.spotify_genre != '']


# Genre histogram
genresJoined = joinedGenres.groupby(['spotify_genre'])['SongID'].count().reset_index() \
                    .sort_values(by=['SongID'], ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(np.arange(30), genresJoined['SongID'].iloc[0:30])
ax.set_xticks(np.arange(30))
ax.set_xticklabels(genresJoined['spotify_genre'][0:30], rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel("Genre")
fig.tight_layout()
fig.suptitle("Frequency of Genres of Billboard Songs", fontsize=20)
fig.subplots_adjust(top=0.9)
fig.savefig("images/genresJoined.png")


# Genre histogram (Unique)
genres = featureGenres.groupby(['spotify_genre'])['SongID'].count().reset_index() \
                    .sort_values(by=['SongID'], ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(np.arange(30), genres['SongID'].iloc[0:30])
ax.set_xticks(np.arange(30))
ax.set_xticklabels(genres['spotify_genre'][0:30], rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel("Genre")
fig.tight_layout()
fig.suptitle("Frequency of Genres of Billboard Songs (Unique)", fontsize=20)
fig.subplots_adjust(top=0.9)
fig.savefig("images/genres.png")


# Genre histogram by decade
genresJoinedDecade = joinedGenres.groupby(['spotify_genre', 'Decade'])['SongID'].count().reset_index() \
                    .sort_values(by=['SongID'], ascending=False)
decades = ["1960s", "1970s", "1980s", "1990s", "2000s","2010s"]
fig, axs = plt.subplots(3, 2, figsize=(14, 14))
i = 0
for ax in axs.flat:
    temp = genresJoinedDecade.loc[genresJoinedDecade['Decade'] == decades[i]]
    ax.bar(np.arange(15), temp['SongID'].iloc[0:15])
    ax.set_ylim((0, 24000))
    ax.set_xticks(np.arange(15))
    ax.set_xticklabels(temp['spotify_genre'][0:15], fontsize="large", rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(decades[i], fontsize="large")
    i += 1
fig.tight_layout()
fig.suptitle("Frequency of Genres of Billboard Songs by Decade", fontsize=28)
fig.subplots_adjust(top=0.9)
fig.savefig("images/genresJoinedDecade.png")


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
numericals = [joined.columns.tolist()[1]] + joined.columns.tolist()[11:21]
numericalMetrics = joined[numericals].groupby(['Year']).aggregate(np.nanmean).reset_index()
numericalMetrics = numericalMetrics.rename(columns={'spotify_track_duration_ms': 'trackduration'})
for metric in numericalMetrics.columns.tolist()[1:]:
    fig, ax = plt.subplots()
    ax.plot(numericalMetrics['Year'], numericalMetrics[metric])
    ax.set_xlabel("Year")
    fig.suptitle("Mean {} of Billboard Songs by Year".format(metric.capitalize()), fontsize=14)
    fig.savefig("images/{}.png".format(metric))


# Test all pairs of columns for correlation coefficient R^2 and filter out most relevant ones
correlations = list(itertools.combinations(features.columns.tolist()[6:15], 2))
for t in correlations:
    r2 = stats.pearsonr(featuresScatter[t[0]], featuresScatter[t[1]])[0]
    if abs(r2) > 0.15:
        print("\nR^2 of " + t[0] + " and " + t[1] + " is " + str(r2))

