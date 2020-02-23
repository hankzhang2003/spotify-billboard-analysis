import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from pymongo import MongoClient
import pprint
from bs4 import BeautifulSoup


client = MongoClient("localhost", 27017)
