# Analysis of Spotify Billboard Hot 100 Songs

# By: Hank Zhang

## Overview

This is an an analysis of the Spotify features of songs on the Billboard Hot 100 Songs chart.  The objective of this project is to analyze the evolution of music over the years

## Description of Dataset

The dataset contains information about the Billboard Hot 100 Songs chart between August 1958 and January 2020.  The data was taken from data.world.  The first data table (hot-stuff.csv, also known as weeks) contains data about the position of each song on the chart over the time period and has 320,000 rows.  The second data table (hot-100-audio features.csv, also known as features) contains data about the features of each song and has 28,500 rows.

&nbsp;

## Columns

### Weeks

* **URL:** The URL of the list of Billboard Hot 100 Songs for that week.
* **WeekID:** A string containing the date at which each measurement of the Billboard Hot 100 Songs was taken.
* **Week position:** The position of a song on the chart for that week.
* **Song:** Title of song.
* **SongID:** A unique ID for each song created from concatenating the song name and artist name.
* **Instance:** The number of different times a song has appeared on the chart.  Used to separate breaks for a given song.
* **Previous week:** The position of the song on the chart the week before.
* **Peak position:** The peak position of a song on the chart.
* **Weeks on chart:** The number of weeks a song has stayed on the chart.

### Features

* **SongID:** A unique ID for each song created from concatenating the song name and artist name.
* **Performer:** Name of artist.
* **Song:** Title of song.
* **Spotify genre:** A list containing all the genres a track is classified under.
* **Spotify track ID:** The ID of a track in Spotify's database.
* **Spotify track album:** Name of album.
* **Spotify track explicit:** Whether a track is considered explicit.
* **Spotify track duration ms:** The length of the track in milliseconds.
* **Spotify track popularity:** A measure of how popular a track is, based on the total number of plays and recency of plays.  Range [0, 100].
* **Danceability:** A measure of how suitable a track is for dancing based on a combination of elements including tempo, rhythm stability, beat strength, and overall regularity.  Range [0, 1].
* **Energy:** A measure of intensity and activity, manifested in features such as dynamic range, perceived loudness, timbre, onset rate, and general entropy. Range [0, 1].
* **Key:** The estimated overall key of the track, using standard pitch class notation.  0 = C, 1 = C\#/D♭, 2 = D,..., 11 = B. 
* **Loudness:** The overall loudness of a track in decibels (dB), averaged across the entire track.  It corresponds to the amplitude of sound waves.  Loudness is measured on a logarithmic scale, and more positive numbers mean louder sounds.  Range [=-60, 0].
* **Mode:** The modality of a track.  1 = major, 0 = minor.
* **Speechiness:** The presence and proportion of spoken words in a track.  Range [0, 1].
* **Acousticness:** A confidence measure of whether the track is acoustic.  Range [0, 1].  Values close to 1 represent close to entirely acoustic tracks.
* **Instrumentalness:** A confidence measure of whether a track contains no vocals.  Range [0, 1].  Values above 0.5 represent roughly instrumental tracks, and values close to 1.0 represent tracks with basically no vocals.
* **Liveness:** A measure that detects the presence of an audience in the recording.  Range [0, 1].  Values above 0.8 represent strong likelihood that the track is a live version rather than a studio version.
* **Valence:** A measure that describes the musical positiveness conveyed by a track.  Tracks with high valence sound more positive (happy, cheerful, euphoric), while tracks with low valence sound more negative (sad, depressed, angry).  Range [0, 1].
* **Tempo:** The overall estimated speed of a track in beats per minute (BPM).  Corresponds to the average beat duration within the track.
* **Time signature:** An estimate of how many beats are in each measure.  Also known as meter.

### Edits

The columns "url", "Instance", "Previous Week Position", "Peak Position", "Weeks on Chart", "spotify_track_id", "spotify_track_preview_url", 
"spotify_track_album", and "spotify_track_popularity" were dropped as they were not needed for this analysis.

The week ID from the weeks table was parsed into a datetime during the import.  Two new columns named Year and Decade were created; Year is the year gathered from the datetime and Decade is a string that describes the decade of that year, obtained by passing the year into a custom-made function.

The spotify track duration was given in milliseconds.  A new column named Track_duration was created that contains the duration in seconds.

Since the spotify genre column in the features table had the genres in the form of a list, the dataframe was stripped of its endings, expanded with explode(), and stripped again of quotes in order to analyze the individual genres of each song and the frequency of each genre.

&nbsp;

## Plots

### Frequency of Genres

![Genres](/images/genresJoined.png)

This graph shows the frequency of the top 30 genres over the time period.  Songs that have remained on the chart for multiple weeks are counted multiple times.  This seems well balanced, with all different kinds of genres represented, from rock to pop to country to hip hop.  This seems consistent with the evolution of music from the 1950s until now.

### Frequency of Genres (Unique)

![Genres unique](/images/genres.png)

This graph shows the frequency of the top 30 genres over the time period.  This set is of the unique songs, so each song only appears once no matter how long it has been on the chart. This is similar to the last graph; the adjustments in the rankings depends on how long each song has remained on the chart.  A genre that is pushed down means that it has stayed on the chart for a long time and has its numbers inflated by multiple submissions.

### Frequency of Genres by Decade

![Genres by decade](/images/genresJoinedDecade.png)

This graph shows the frequency of the top 15 genres during each decade.  Here one can see the clear evolution of the popularity of certain genres.  Most notably, genres like rock and soul were quite popular up until the 1980s.  In the 1990s and 2000s, pop, dance, and hip hop started taking over as the most popular genres. 

### Explicitness

![Explicitness](/images/explicitness.png)

This graph shows the proportion of explicit songs in a time series over the time period.  What is interesting is that there is little to no presence of profanity in music up until 1990.  However, the amount of tracks with explicit material increased sharply after that time period and takes an even more alarming leap around 2015. This can partially be attributed to the increase in hip hop and rap music which tend to have words and topics that are significantly more on the explicit side, whereas older music tends to have more mellow topics and lyrics.

### Track duration

![Track duration](/images/track_duration.png)

This graph shows the mean track duration (seconds) of songs in a time series over the time period.  There is an increase around the 1970s followed by a decrease in the 1990s but overall there is no real trend that evolves over time.

### Danceability, Energy, Loudness

![Danceability](/images/danceability.png)
![Energy](/images/energy.png)
![Loudness](/images/loudness.png)

These graphs show the mean danceability, energy, and loudness (dB) of songs in a time series over the time period.  These are typically characteristics of upbeat songs.  Over the past few decades the music people listen to has gotten significantly more upbeat than before, resulting in similar upward trends of these metrics.

### Speechiness and Instrumentalness

![Speechiness](/images/speechiness.png)
![Instrumentalness](/images/instrumentalness.png)

These graphs show the mean speechiness and instrumentalness of songs in a time series over the time period.  These are 2 metrics that typically contrast with one another, as more vocals in a song typically means fewer instrumental parts.

### Acousticness

![Acousticness](/images/acousticness.png)

This graph shows the mean acousticness of songs in a time series over the time period.  As expected, the acousticness of songs goes down over time due to the release of new technologies and hence more encouragement to produce more electronic music rather than using traditional instruments.

### Liveness

![Liveness](/images/liveness.png)

This graph shows the mean liveness of songs in a time series over the time period.  This has also decreased over the decades, but the trend has gone up down a lot rather than remaining homogeneous.  Most live tracks come from concerts, which are typically held by bands, and the genres bar charts have already shown that the genres of music most likely to be played by bands peaked in popularity in the 1960s and 1980s. 

### Valence

![Valence](/images/valence.png)

This graph shows the mean valence of songs in a time series over the time period.  The average valence of songs has actually gone down over the years, indicating an overall decrease of positiveness in the music.  This is a difficult metric to measure numerically and more exploration and data analysis techniques will be needed to make a definitive conclusion about valence and positiveness.

### Tempo

![Tempo](/images/tempo.png)

This graph shows the mean tempo of songs in a time series over the time period.  Interestingly, since the danceability, energy, and loudness of songs had a definitive increase, one would expect tempo to have an increase as well.  However, the average tempo had little net change over the years; it went up and down many times.

### Energy and Loudness

![Energy and Loudness](/images/energyandloudness.png)
![Energy vs Loudness](/images/energyvsloudnessScatter.png)

asdf

### Energy and Danceability

![Energy and Danceability](/images/energyanddanceability.png)
![Energy and Danceability](/images/energyvsdanceabilityScatter.png)

asdf


### Acousticness and Energy

![Acousticness and Energy](/images/acousticnessandenergy.png)
![Acousticness vs Energy](/images/acousticnessvsenergyScatter.png)

asdf

### Acousticness and Loudness

![Acousticness and Loudness](/images/acousticnessandloudness.png)
![Acousticness vs Loudness](/images/acousticnessvsloudnessScatter.png)

asdf

### Energy and Valence

![Energy and Valence](/images/energyandvalence.png)
![Energy vs Valence](/images/energyvsvalenceScatter.png)

asdf

### Danceability and Valence

![Danceability and Valence](/images/danceabilityandvalence.png)
![Danceability vs Valence](/images/danceabilityvsvalenceScatter.png)

asdf

### Energy and Tempo

![Energy and Tempo](/images/energyandtempo.png)
![Energy vs Tempo](/images/energyvstempoScatter.png)

asdf








