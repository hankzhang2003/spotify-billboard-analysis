import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import csvs
features = pd.read_csv("data/hot-100-audio-features.csv", converters= {'artist_genre':
                        lambda s: s[1:-1].split(', ')}, encoding='latin-1')
features['artist_genre'] = features['artist_genre'].apply(lambda l: [s[1:-1] for s in l])
weeks = pd.read_csv("data/hot-stuff.csv")
print(features.head())
print(weeks.head())

# Join tables together
joined = weeks.merge(features, on='SongID')
print(joined.head())

# Expand genres
featureGenres = features.explode('artist_genre')
print(featureGenres.head())


