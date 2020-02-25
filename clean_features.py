import numpy as np
import pandas as pd


# Functions to import and clean data

def clean_features() -> pd.DataFrame:
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
    features['spotify_genre'] = features['spotify_genre'].map(lambda l: [s[1:-1] for s in l])
    features = features.dropna().drop_duplicates("SongID").reset_index(drop=True)
    features = features[features['tempo'] != 0]
    features['spotify_track_explicit'] = features['spotify_track_explicit'].astype(float)
    features['spotify_track_duration_ms'] = features['spotify_track_duration_ms'] / 1000
    features.rename(columns={"spotify_track_duration_ms": "track_duration"}, inplace=True)
    return features
