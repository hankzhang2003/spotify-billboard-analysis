import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import string
import ssl
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup


def parse_page(title: str, artist: str):
    titleFixed = title.lower().replace("&", "and").translate(str.maketrans(" ", "-", string.punctuation))
    artistFixed = artist.lower().replace("&", "and").translate(str.maketrans(" ", "-", string.punctuation))
    sep = "-feat"
    artistFixed = artistFixed.split(sep, 1)[0]
    artistFixed = artistFixed.replace("--", "-")
    url = "https://genius.com/{}-{}-lyrics".format(artistFixed, titleFixed)
    try:
        req = Request(url, headers = {"User-Agent": "Mozilla/73.0"})
        webpage = urlopen(req).read()
        soup = BeautifulSoup(webpage, "html.parser")
        html = soup.prettify("utf-8")
        for div in soup.find_all("div", attrs = {"class": "lyrics"}):
            songLyrics = div.text.strip().split("\n")
        return songLyrics
    except:
        #print(titleFixed, ", ", artistFixed, ", ", url)
        return ["*********************", "Error: URL not valid", titleFixed, artistFixed, url]
