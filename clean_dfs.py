import numpy as np
import pandas as pd


# Functions to import and clean data

# Read and clean features
def clean_features() -> pd.DataFrame:
    features = pd.read_csv("data/hot-100-audio-features.csv", converters={'spotify_genre':
                            lambda s: s[1:-1].split(", ")}, encoding="latin-1")
    featuresFilter = ["spotify_track_id", "spotify_track_preview_url", "spotify_track_album",
                      "spotify_track_popularity", "key", "time_signature"]
    features.drop(featuresFilter, axis=1, inplace=True)
    features = features[[len(features['spotify_genre'][i][0]) != 0 for i in range(len(features['spotify_genre']))]]
    features['spotify_genre'] = features['spotify_genre'].map(lambda l: [s[1:-1] for s in l])
    features = features.dropna().drop_duplicates("SongID").reset_index(drop=True)
    features = features[features['tempo'] != 0]
    features['spotify_track_explicit'] = features['spotify_track_explicit'].astype(float)
    features['spotify_track_duration_ms'] = features['spotify_track_duration_ms'] / 1000
    features.rename(columns={"spotify_track_duration_ms": "track_duration"}, inplace=True)
    return features

# Read and clean weeks
def clean_weeks() -> pd.DataFrame:
    weeks = pd.read_csv("data/hot-stuff.csv", converters={'WeekID': lambda d:
                        pd.to_datetime(d, format="%m/%d/%Y", errors="coerce")})
    weeksFilter = ["url", "Instance", "Previous Week Position", "Peak Position",
                   "Weeks on Chart"]
    weeks = weeks.drop(weeksFilter, axis=1).drop_duplicates().reset_index(drop=True)
    if "Year" not in weeks.columns:
        weeks.insert(1, "Year", weeks['WeekID'].dt.year)
    if "Decade" not in weeks.columns:
        weeks.insert(2, "Decade", weeks['Year'].apply(decade))
    return weeks

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

# Read csv of previously outputted scraped lyrics and reformat to match original
def clean_lyrics() -> pd.DataFrame:
    lyrics = pd.read_csv("data/scrapedLyrics.csv", converters={'Lyrics': lambda s: 
                            s[1:-1].split(", ")})
    lyrics['Lyrics'] = lyrics['Lyrics'].map(lambda l: [s[1:-1] for s in l])
    lyrics['Lyrics'] = lyrics['Lyrics'].map(lambda l: [s.replace("\\", "") for s in l])
    lyrics = lyrics[[valid_lyrics(l) for l in lyrics['Lyrics']]]
    lyrics['Lyrics'] = list(map(clean_line, lyrics['Lyrics']))
    lyrics.reset_index(drop=True, inplace=True)
    return lyrics

# Remove all invalid lines in scraped lyrics and join into 1 string
def clean_line(lyrics: list) -> str:
    cleanedLyrics = [line for line in lyrics if len(line) != 0 and line[0] != "(" \
                        and line[0] != "[" and line[0] != "{"]
    return " ".join(cleanedLyrics)

def valid_lyrics(lyrics: str) -> bool:
    return lyrics[0][0] != "*"
