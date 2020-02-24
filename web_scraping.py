import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import ssl
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup


ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def parse_page(title: str, artist: str):
    url = "https://genius.com/{}-{}-lyrics".format(artist.replace(" ", "-"), title.lower().replace(" ", "-"))
    req = Request(url, headers = {"User-Agent": "Mozilla/5.0"})
    webpage = urlopen(req).read()
    soup = BeautifulSoup(webpage, "html.parser")
    html = soup.prettify("utf-8")
    lyrics = []
    for div in soup.findAll('div', attrs = {'class': 'lyrics'}):
        lyrics.append(div.text.strip().split("\n"));
    return lyrics

test = parse_page("Dance the Night Away", "Twice")
test2 = parse_page("7 rings", "Ariana Grande")
