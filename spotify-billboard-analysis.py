import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import csvs and remove unnecessary columns
weeks = pd.read_csv("data/hot-stuff.csv", converters={'WeekID': lambda d: pd.to_datetime(d, \
                    format="%m/%d/%Y", errors="coerce")})
weeksFilter = ['url', 'Instance', 'Previous Week Position', 'Peak Position',
               'Weeks on Chart']
weeks.drop(weeksFilter, axis=1, inplace=True)

features = pd.read_csv("data/hot-100-audio-features.csv", converters={'spotify_genre':
                        lambda s: s[1:-1].split(', ')}, encoding="latin-1")
features['spotify_genre'] = features['spotify_genre'].apply(lambda l: [s[1:-1] for s in l])
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

joined = weeks.merge(features, on='SongID')
joined.to_csv("data/joined.csv", index=False)

featureGenres = features.explode('spotify_genre')
featureGenres = featureGenres[featureGenres.spotify_genre != ''].dropna()

joinedGenres = joined.explode('spotify_genre')
joinedGenres = joinedGenres[joinedGenres.spotify_genre != ''].dropna()


# Genre histogram
genresJoined = joinedGenres.groupby(['spotify_genre'])['SongID'].count().reset_index() \
                    .sort_values(by=['SongID'], ascending=False)
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(30), genresJoined['SongID'].iloc[0:30])
ax.set_xticks(range(30))
ax.set_xticklabels(genresJoined['spotify_genre'][0:30], rotation=45)
ax.set_xlabel("Genre")
fig.suptitle("Frequency of Genres of Billboard Hot 100 Songs", fontsize=20)
fig.savefig("images/genresJoined.png")


# Genre histogram (Unique)
genres = featureGenres.groupby(['spotify_genre'])['SongID'].count().reset_index() \
                    .sort_values(by=['SongID'], ascending=False)
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(30), featureGenres['SongID'].iloc[0:30])
ax.set_xticks(range(30))
ax.set_xticklabels(featureGenres['spotify_genre'][0:30], rotation=45)
ax.set_xlabel("Genre")
fig.suptitle("Frequency of Genres of Billboard Hot 100 Songs (Unique)", fontsize=20)
fig.savefig("images/genres.png")


# Genre histogram by decade
genresJoinedDecade = joinedGenres.groupby(['spotify_genre', 'Decade'])['SongID'].count().reset_index() \
                    .sort_values(by=['SongID'], ascending=False)
decades = ["1960s", "1970s", "1980s", "1990s", "2000s","2010s"]
i = 0
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
for ax in axs.flat:
    temp = genresJoinedDecade.loc[genresJoinedDecade['Decade'] == decades[i]]
    ax.bar(range(10), temp['SongID'].iloc[0:10])
    ax.set_xticks(range(10))
    ax.set_xticklabels(temp['spotify_genre'][0:10], rotation=45)
    ax.set_title(decades[i])
    i += 1
fig.tight_layout()
fig.suptitle("Frequency of Genres of Billboard Hot 100 Songs by Decade", fontsize=24)
fig.subplots_adjust(top=0.9)
fig.savefig("images/genresJoinedDecade.png")


# Mean of each numerical metric by year
numericalsYear = [joined.columns.tolist()[1]] + joined.columns.tolist()[11:21]
numericalMetricsYear = joined[numericalsYear].groupby(['Year']).mean().reset_index()
numericalMetricsYear = numericalMetricsYear.rename(columns={'spotify_track_duration_ms': 'trackduration'})
for metric in numericalMetricsYear.columns.tolist()[1:]:
    fig, ax = plt.subplots()
    ax.plot(np.arange(1958, 2020), numericalMetricsYear[metric])
    ax.set_xlabel("Year")
    fig.suptitle("Mean {} of Billboard Hot 100 Songs by Year".format(metric.capitalize()), fontsize=12)
    fig.savefig("images/{}Year.png".format(metric))


