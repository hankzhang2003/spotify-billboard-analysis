import numpy as np
import pandas as pd


# Create binary classfier for each genre group (rock, pop, hip hop/rap, jazz, etc)
def get_bucket(genre: str, buckets: dict) -> int:
    for key, value in buckets.items():
        if genre in value:
            return key
    return float("nan")

# Returns true if song has any of those genre types
def contains_genre_type(genre_list: list, genre_types: list) -> bool:
    for g in genre_list:
        for t in genre_types:
            if t in g:
                return True
    return False

# Create binary column that can be added to dataframe
def create_genre_column(genre_column: list, genre_types: str) -> list:
    return [int(contains_genre_type(g, genre_types)) for g in genre_column]
