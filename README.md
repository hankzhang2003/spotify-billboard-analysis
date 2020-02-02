# Analysis of Spotify Billboard Hot 100 Songs

# By: Hank Zhang

## Overview

This is an an analysis of the Spotify features of songs on the Billboard Hot 100 Songs chart.  The objective of this project is to perform exploratory data analysis on the Spotify trends of popular music over the past 60 years.  There are many categorical and numerical varables with characteristics of each song, from basic metrics such as title, artist, and album, to metrics that may not even properly expressible with numbers such as energy, danceability, and valence.

&nbsp;

## Description of Dataset

The dataset contains information about the Billboard Hot 100 Songs chart between August 1958 and January 2020.  This dataset was taken from data.world but all of this data is obtainable by scraping Spotify's API using Selenium and BeautifulSoup.  The first data table (hot-stuff.csv, also known as weeks) contains data about the position of each song on the chart over the time period and has 320,000 rows.  The second data table (hot-100-audio features.csv, also known as features) contains data about the features of each song and has 28,500 rows.  There are both categorical and numerical variables.

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

Since the spotify genre column in the features table had the genres in the form of a list, the dataframe was stripped of its endings, expanded with explode(), and stripped again of quotes in order to analyze the individual genres of each song and the frequency of each genre.  The resulting data frame was 1.4 million rows when expanded.

### Data types (features)

Categorical: URL, weekID, year, decade, song, songID

Numeric: week position, instance, previous week, peak position, weeks on chart

### Data types (weeks)

Categorical: songID, performer, song, spotify genre, spotify track ID, spotify track album, spotify track explicit, mode, time signature

Numeric: spotify track duration ms, track duration, spotify track popularity, danceability, key, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo

&nbsp;

## Plots and Analysis

### Frequency of Genres

![Genres](/images/genresJoined.png)

This graph shows the frequency of the top 30 genres over the time period.  Songs that have remained on the chart for multiple weeks are counted multiple times.  This seems well balanced, with all different kinds of genres represented, from rock to pop to country to hip hop.  This seems consistent with the evolution of music from the 1950s until now.

### Frequency of Genres (Unique)

![Genres unique](/images/genres.png)

This graph shows the frequency of the top 30 genres over the time period.  This set is of the unique songs, so each song only appears once no matter how long it has been on the chart. This is similar to the last graph; the adjustments in the rankings depends on how long each song has remained on the chart.  A genre that is pushed down means that it has stayed on the chart for a long time and has its numbers inflated by multiple submissions.

### Frequency of Genres by Decade

![Genres by decade](/images/genresJoinedDecade.png)

This graph shows the frequency of the top 15 genres during each decade.  Here one can see the clear evolution of the popularity of certain genres.  Most notably, genres like rock and soul were quite popular up until the 1980s.  In the 1990s and 2000s, pop, dance, and hip hop started taking over as the most popular genres. 

&nbsp;

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

These graphs show the mean danceability, energy, and loudness (decibels) of songs in a time series over the time period.  These are typically characteristics of upbeat songs.  Over the past few decades the music people listen to has gotten significantly more upbeat than before, resulting in similar upward trends of these metrics.  The most notable changes were around 1980 and 2000.

### Speechiness and Instrumentalness

![Speechiness](/images/speechiness.png)
![Instrumentalness](/images/instrumentalness.png)

These graphs show the mean speechiness and instrumentalness of songs in a time series over the time period.  These are 2 metrics that typically contrast with one another, as more vocals in a song typically means fewer instrumental parts.  As one increases, the other decreases.  The most notable change was in the 1990s.

### Acousticness

![Acousticness](/images/acousticness.png)

This graph shows the mean acousticness of songs in a time series over the time period.  As expected, the acousticness of songs has a definite decrease due to the release of new technologies and hence more encouragement to produce more electronic music rather than using traditional instruments like before.

### Liveness

![Liveness](/images/liveness.png)

This graph shows the mean liveness of songs in a time series over the time period.  This has also decreased over the decades, but the trend has gone up down a lot rather than remaining homogeneous.  Most live tracks come from concerts, which are typically held by bands, and the genres bar charts have already shown that the genres of music most likely to be played by bands peaked in popularity in the 1960s and 1980s. 

### Valence

![Valence](/images/valence.png)

This graph shows the mean valence of songs in a time series over the time period.  The average valence of songs has actually gone down over the years, indicating an overall decrease of positiveness in the music.  This is a difficult metric to measure numerically and more exploration and data analysis techniques will be needed to make a definitive conclusion about valence and positiveness.

### Tempo

![Tempo](/images/tempo.png)

This graph shows the mean tempo (beats per minute) of songs in a time series over the time period.  Interestingly, since the danceability, energy, and loudness of songs had a definitive increase, one would expect tempo to have an increase as well.  However, the average tempo had little net change over the years; it went up and down many times but only ended up a little bit above the initial values in the 1950s.

&nbsp;

### Energy and Loudness

![Energy and Loudness](/images/energyandloudness.png)
![Energy vs Loudness](/images/energyvsloudnessScatter.png)

These graphs compare energy and loudness, both as a function of time and against each other.  There is a definitive positive correlation between the two variables.  The Pearson r<sup>2</sup> correlation coefficient is 0.6861.  This makes sense because as more energy is put into a song, it also tends to be louder and push up the decibel level.

### Energy and Danceability

![Energy and Danceability](/images/energyanddanceability.png)
![Energy and Danceability](/images/energyvsdanceabilityScatter.png)

These graphs compare energy and danceability, both as a function of time and against each other.  There is a slight positive correlation between the two variables.  The Pearson r<sup>2</sup> correlation coefficient is 0.2044.  This was slightly surprising since I expected the correlation to be higher since logically one of the key factors in seeing whether a song is danceable is how much energy it has.

### Acousticness and Energy

![Acousticness and Energy](/images/acousticnessandenergy.png)
![Acousticness vs Energy](/images/acousticnessvsenergyScatter.png)

These graphs compare acousticness and energy, both as a function of time and against each other.  There is a definitive negative correlation between the two variables.  The Pearson r<sup>2</sup> correlation coefficient is -0.5875.  This was slightly surprising since I did not expect the correlation to be that high, since these 2 metrics seem to only be loosely related in the big picture and other metrics seem like they would be more closely related.

### Acousticness and Loudness

![Acousticness and Loudness](/images/acousticnessandloudness.png)
![Acousticness vs Loudness](/images/acousticnessvsloudnessScatter.png)

These graphs compare acousticness and loudness, both as a function of time and against each other.  There is a definitive negative correlation between the two variables.  The Pearson r<sup>2</sup> correlation coefficient is -0.4059.  Similar to the previous graph, this makes sense because as a song because as there is more acousticness and less electronic-ness, there is less potential to create loud sounds.

### Energy and Valence

![Energy and Valence](/images/energyandvalence.png)
![Energy vs Valence](/images/energyvsvalenceScatter.png)

These graphs compare energy and valence, both as a function of time and against each other.  There is a slight positive correlation between the two variables.  The Pearson r<sup>2</sup> correlation coefficient is 0.3560.  This makes sense because high energy songs tend to elicit higher levels of positive feelings.

### Danceability and Valence

![Danceability and Valence](/images/danceabilityandvalence.png)
![Danceability vs Valence](/images/danceabilityvsvalenceScatter.png)

These graphs compare danceability and valence, both as a function of time and against each other.  There is a definitive positive correlation between the two variables.  The Pearson r<sup>2</sup> correlation coefficient is 0.3963.  Similar to the previous graph, this makes sense because songs that people are more likely to dance to also elicit higher levels of positive feelings.

### Energy and Tempo

![Energy and Tempo](/images/energyandtempo.png)
![Energy vs Tempo](/images/energyvstempoScatter.png)

These graphs compare energy and tempo, both as a function of time and against each other.  There is a slight positive correlation between the two variables.  The Pearson r<sup>2</sup> correlation coefficient is 0.1620.  This was slightly surprising since I expected the correlation to be higher since logically a high or low energy level in a track would be reflected in how fast the song is.

&nbsp;

## Hypothesis Testing

Hypothesis testing was done on the metrics that were closest to normally distributed, which were energy, danceability, loudness, valence, and tempo.  The two-sample t-test and Mann-Whitney U-test were both used.  The two-sample t-test is the standard hypothesis testing method used to compare two population means.  The Mann-Whitney U-test was used as a backup in case the original distribution was too differet from a normal distribution and hence the z-test and t-test would be inaccurate.

### Hypothesis Test Setup

H<sub>0</sub> = Music today has the same energy, danceability, loudness, valence, and tempo as music of 60 years ago.

H<sub>a</sub> = Music today does not have the same energy, danceability, loudness, valence, and tempo as music of 60 years ago.

&alpha; = 0.05, can be set to 0.02 if we want an even higher level of certainty.

### Two-sample t-test Results

Energy: u = -138.0755, p = 0.0

Danceability: u = -121.4532, p = 0.0

Loudness: u = -245.8457, p = 0.0

Valence: u = 114.6155, p = 0.0

Tempo: u = -14.3916, p = 6.559112204592384e-47

### Mann-Whitney U-test Results

Energy: u = 5.6794056e8, p = 0.0

Danceability: u = 6.2678145e8, p = 0.0

Loudness: u = 2.5575936e8, p = 0.0

Valence: u = 1.5164469e9, p = 0.0

Tempo: u = 1.0079646e9, p = 2.7332407119580174e-62

&nbsp;

All of the p-values are far too low for both tests, so we reject every single null hypothesis.  Therefore, we can accept the alternative hypotheses and conclude that there is a statistically significant difference between the features of the music of the 1960s and 2010s.

## Conclusion

This analysis is just an elementary EDA of the features of the Spotify Billboard Hot 100 dataset, and there are many other insights and data analysis techniques one can use on these expansive tables.  One possible direction is to compare the data from the United States to the data of another country, such as the Billboard Korea K-Pop 100, and see the differences in the audio features between the popular songs in each country.  It would be helpful to see what people from different countries like in their music.  Another possible direction is to group genres into several clusters and predict the overall genre of a song given its set of features.  This would be helpful for categorizing songs based on their features and recommending songs to people, in a manner similar to Spotify Radio or a "Songs You May Like" list.

Music is an aspect of culture and life that has existed since the dawn of mankind.  Over the years and ages, music has evolved from primitive instruments such as logs and rocks to old-school rock bands and jazz to modern hip hop and electronic music.  At its core, most tracks can be broken down into a set of features that can be represented with either words or numbers.  The ability to analyze the trends of music is only one of many factors in the analysis of human cultural evolution itself.
