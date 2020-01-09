# Analysis of Spotify Billboard Hot 100 Songs

# By: Hank Zhang

## Description of Dataset

This dataset contains information about the Billboard Hot 100 Songs chart between August 1958 and January 2020.  The data was taken from data.world.  The first data table (hot-stuff.csv, also known as weeks) contains data about the position of each song on the chart over the time period and has 320,000 rows.  The second data table (hot-100-audio features.csv, also known as features) contains data about the features of each song and has 28,500 rows.

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

Since the spotify genre column in the features table had the genres in the form of a list, the dataframe was stripped of its endings, expanded with explode(), and stripped again of quotes in order to analyze the individual genres of each song and the frequency of each genre.

&nbsp;

## Plots

### Frequency of Genres

![Genres](/images/genresJoined.png)

This graph shows the frequency of the top 30 genres over the time period.  Songs that have remained on the chart for multiple weeks are counted multiple times.  This seems well balanced, with all different kinds of genres represented, from rock to pop to 


### Frequency of Genres (Unique)

![Genres unique](/images/genres.png)

This graph shows the frequency of the top 30 genres over the time period.  This set is of the unique songs, so each song only appears once no matter how long it has been on the chart.

### Frequency of Genres by Decade

![Genres by decade](/images/genresJoinedDecade.png)

This gra

### Explicitness

![Explicitness](/images/explicitness.png)

This graph shows the proportion of explicit tracks over the time period.  What is interesting is that there is little to no presence of profanity in music up until 1990.  However, the amount of tracks with explicit material increased sharply after that time period and takes an even more alarming leap around 2015.

### Track duration

![Track duration](/images/trackduration.png)

This graph shows the mean track duration of songs over the time period.  There is an increase around the 1970s but overall there is no real trend that evolves over time.

### Danceability

![Danceability](/images/danceability.png)

asdf

### Energy

![Energy](/images/energy.png)

asdf

### Loudness

![Loudness](/images/loudness.png)

asdf

### Speechiness

![Speechiness](/images/speechiness.png)

asdf

### Acousticness

![Acousticness](/images/acousticness.png)

asdf

### Instrumentalness

![Instrumentalness](/images/instrumentalness.png)

asdf

### Liveness

![Liveness](/images/liveness.png)

asdf

### Valence

![Valence](/images/valence.png)

asdf

### Tempo

![Tempo](/images/tempo.png)

asdf

### Acousticness and Energy

![Acousticness and Energy](/images/acousticnessandenergy.png)

asdf

### Energy and Loudness

![Energy and Loudness](/images/energyandloudness.png)

asdf

### Acousticness and Loudness

![Acousticness and Loudness](/images/acousticnessandloudness.png)

asdf

### Energy and Danceability

![Energy and Danceability](/images/energyanddanceability.png)

asdf

### Danceability and Valence

![Danceability and Valence](/images/danceabilityandvalence.png)

asdf

### Acousticness vs Energy

![Acousticness vs Energy](/images/acousticnessvsenergyScatter.png)

asdf

### Energy vs Loudness

![Energy vs Loudness](/images/energyvsloudnessScatter.png)

asdf

### Acousticness vs Loudness

![Acousticness vs Loudness](/images/acousticnessvsloudnessScatter.png)

asdf

### Energy vs Danceability

![Energy and Danceability](/images/energyvsdanceabilityScatter.png)

asdf

### Danceability vs Valence

![Acousticness vs Energy](/images/danceabilityvsvalenceScatter.png)

asdf







