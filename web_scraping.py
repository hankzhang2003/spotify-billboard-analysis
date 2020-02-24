import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import ssl
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup


def parse_page(title: str, artist: str):
    titleFixed = title.lower().replace(" ", "-")
    artistFixed = artist.lower().replace("&", "and").replace(" ", "-").replace(",", "")
    sep = "-feat"
    artistFixed = artistFixed.split(sep, 1)[0]
    url = "https://genius.com/{}-{}-lyrics".format(artistFixed, titleFixed)
    try:
        req = Request(url, headers = {"User-Agent": "Mozilla/73.0"})
        webpage = urlopen(req).read()
        soup = BeautifulSoup(webpage, "html.parser")
        html = soup.prettify("utf-8")
        songLyrics = []
        for div in soup.findAll("div", attrs = {"class": "lyrics"}):
            songLyrics.append(div.text.strip().split("\n"))
        songLyrics = songLyrics[0]
        return songLyrics
    except:
        return ["Error: URL not valid", titleFixed, artistFixed, url]
