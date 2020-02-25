import numpy as np
import pandas as pd


# Create binary classfier for each genre group (rock, pop, hip hop/rap, jazz, etc)
def get_bucket(genre: str, buckets: dict) -> int:
    for key, value in buckets.items():
        if genre in value:
            return key
    return float("nan")

# Returns true if song has that genre type
def contains_genre_type(genre_list: list, genre_type: str) -> bool:
    for g in genre_list:
        if genre_type in g:
            return True
    return False

# Add column to data frame
def create_genre_column(genre_column: list, genre_type: str) -> list:
    return [int(contains_genre_type(g, genre_type)) for g in genre_column]
