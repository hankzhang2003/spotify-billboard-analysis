# Analysis of Spotify Billboard Hot 100 Songs

## By: Hank Zhang

### Description of Dataset

This dataset contains information about the Billboard Hot 100 Songs chart ranging from 8/2/1958 to 6/22/2019.  The data was taken from data.world.  The first data table contains data about the position of each song on the chart over the time period and has 318,000 rows.  The second data table contains data about the features of each song and has 28,000 rows.

### Columns

* **URL:** The URL of the list of Billboard Hot 100 Songs for that week.
* **WeekID:** A string containing the date at which each measurement of the Billboard Hot 100 Songs was taken.
* **Week position:** The position of a song on the chart for that week.
* **Song:** Title of song.
* **Performer:** Name of artist.
* **SongID:** A unique ID for each song created from concatenating the song name and artist name.
* **Instance:** The number of different times a song has appeared on the chart.  Used to separate breaks for a given song.
* **Previous week:** The position of the song on the chart the week before.
* **Peak position:** The peak position of a song on the chart.
* **Weeks on chart:** The number of weeks a song has stayed on the chart.

* **Spotify track ID:** The ID of a track in Spotify's database.
* **Artist genre:** A list containing all the genres a track is classified under.
* **Spotify track duration ms:** The length of the track in milliseconds.
* **Spotify track popularity:**
* **Danceability:** A measure of how suitable a track is for dancing based on a combination of elements including tempo, rhythm stability, beat strength, and overall regularity.  Range [0, 1].
* **Energy:** A measure of intensity and activity, manifested in features such as dynamic range, perceived loudness, timbre, onset rate, and general entropy. Range [0, 1].
* **Key:** The estimated overall key of the track, using standard pitch class notation.  0 = C, 1 = C\#/D♭, 2 = D,..., 11 = B. 
* **Loudness:** The overall loudness of a track in decibels (dB), averaged across the entire track.  Corresponds to the amplitude of sound waves.  Range [=-60, 0].
* **Mode:** The modality of a track.  1 = major, 0 = minor.
* **Speechiness:** The presence and proportion of spoken words in a track.  Range [0, 1].
* **Acousticness:** A confidence measure of whether the track is acoustic.  Range [0, 1].  Values close to 1 represent close to entirely acoustic tracks.
* **Instrumentalness:** A confidence measure of whether a track contains no vocals.  Range [0, 1].  Values above 0.5 represent roughly instrumental tracks, and values close to 1.0 represent tracks with basically no vocals.
* **Liveness:** A measure that detects the presence of an audience in the recording.  Range [0, 1].  Values above 0.8 represent strong likelihood that the track is a live version rather than a studio version.
* **Valence:** A measure that describes the musical positiveness conveyed by a track.  Tracks with high valence sound more positive (happy, cheerful, euphoric), while tracks with low valence sound more negative (sad, depressed, angry).  Range [0, 1].
* **Tempo:** The overall estimated speed of a track in beats per minute (BPM).  Corresponds to the average beat duration within the track.
* **Time signature:** An estimate of how many beats are in each measure.  Also known as meter.

