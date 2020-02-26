import numpy as np
import pandas as pd
import string
import ssl
from urllib.request import Request, urlopen
from threading import Thread
from bs4 import BeautifulSoup


def parse_page(title: str, artist: str) -> list:
    titleFixed = title.lower()
    artistFixed = artist.lower()
    if "," in artistFixed:
        artistFixed = artistFixed.split(",", 1)[0]
    if "feat" in artistFixed:
        artistFixed = artistFixed.split("feat", 1)[0]
    titleFixed = titleFixed.replace("&", "and").translate(str.maketrans(" !$/()",
                    "------", "',.?+"))
    artistFixed = artistFixed.replace("&", "and").translate(str.maketrans(" !$/()",
                    "------", "'.?+"))
    titleFixed = titleFixed.replace("--", "-")
    artistFixed = artistFixed.replace("--", "-")
    url = "https://genius.com/{}-{}-lyrics".format(artistFixed, titleFixed)
    url = url.replace("--", "-")
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/73.0"})
        webpage = urlopen(req).read()
        soup = BeautifulSoup(webpage, "html.parser")
        html = soup.prettify("utf-8")
        for div in soup.find_all("div", attrs={"class": "lyrics"}):
            songLyrics = div.text.strip().split("\n")
        return songLyrics
    except:
        # googleurl = "https://www.google.com/asdf"
        # search "{} {} lyrics genius".format(title, artist)
        return ["*******", titleFixed, artistFixed, url]

def store_lyrics(title: str, artist: str, d: dict) -> dict:
    songID = title + artist
    lyrics = parse_page(title, artist)
    lyrics = [line.replace(",", "") for line in lyrics]
    d[songID] = lyrics
    return d

def read_lyrics() -> list:
    lyrics = pd.read_csv("data/scrapedLyrics.csv", converters={'Lyrics': lambda s: 
                            s[1:-1].split(", ")})
    lyrics['Lyrics'] = lyrics['Lyrics'].map(lambda l: [s[1:-1] for s in l])
    lyrics['Lyrics'] = lyrics['Lyrics'].map(lambda l: [s.replace("\\", "") for s in l])
    return lyrics

def clean_lyrics(lyrics: list) -> bool:
    cleanedLyrics = [line for line in lyrics if len(line) != 0 and line[0] != "["]
    return cleanedLyrics