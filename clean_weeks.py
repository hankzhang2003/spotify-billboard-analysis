import numpy as np
import pandas as pd


# Functions to import and clean

def clean_weeks():
    weeks = pd.read_csv("data/hot-stuff.csv", converters={'WeekID': lambda d: pd.to_datetime(d, \
                    format="%m/%d/%Y", errors="coerce")})
    weeksFilter = ["url", "Instance", "Previous Week Position", "Peak Position",
                   "Weeks on Chart"]
    weeks.drop(weeksFilter, axis=1, inplace=True)
    weeks.drop_duplicates(inplace=True)
    weeks.reset_index(inplace=True)
    if "Year" not in weeks.columns:
        weeks.insert(1, "Year", weeks['WeekID'].dt.year)
    if "Decade" not in weeks.columns:
        weeks.insert(2, "Decade", weeks['Year'].apply(decade))
    return weeks

def decade(year: int) -> str:
    if year >= 1950 and year < 1960:
        return "1950s"
    elif year >= 1960 and year < 1970:
        return "1960s"
    elif year >= 1970 and year < 1980:
        return "1970s"
    elif year >= 1980 and year < 1990:
        return "1980s"
    elif year >= 1990 and year < 2000:
        return "1990s"
    elif year >= 2000 and year < 2010:
        return "2000s"
    elif year >= 2010 and year < 2020:
        return "2010s"
    else:
        return None
