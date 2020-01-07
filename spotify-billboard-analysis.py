import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


# Import
features = pd.read_csv("data/hot-100-audio-features.csv", converters={'spotify_genre':
                        lambda s: s[1:-1].split(', ')}, encoding='latin-1')
features['spotify_genre'] = features['spotify_genre'].apply(lambda l: [s[1:-1] for s in l])
weeks = pd.read_csv("data/hot-stuff.csv", converters={'WeekID': lambda d: datetime. \
                    strptime(d,"%m/%d/%Y").date()})
joined = weeks.merge(features, on='SongID')


# Genre histogram
featuresExpanded = features.explode('spotify_genre')
featuresExpanded = featuresExpanded[featuresExpanded.spotify_genre != ''].dropna()
featuresExpanded.to_csv("data/features-expanded.csv")

featureGenres = featuresExpanded.groupby('spotify_genre')['SongID'].count().reset_index() \
                    .sort_values(by=['SongID'], ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(20), featureGenres['SongID'].iloc[0:20])
ax.set_xticks(range(20))
ax.set_xticklabels(featureGenres['spotify_genre'][0:20], rotation=90)
ax.set_title("Frequency of Genres of Billboard Hot 100 Songs")
fig.savefig("images/genre_hist.png")


#

