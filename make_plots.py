import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


# Functions to make plots for EDA

def make_frequency_plot(df: pd.DataFrame, top: int, ax: plt.axes) -> None:
    ax.bar(np.arange(top), df['SongID'].iloc[0:top])
    ax.set_xticks(np.arange(top))
    ax.set_xticklabels(df['spotify_genre'][0:top], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Genre", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)

def make_line_plot(df: pd.DataFrame, col: str, ax: plt.axes) -> None:
    ax.plot(df['Year'], df[col])
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("{}".format(col.capitalize()), fontsize=14)

def make_dual_plot_same(df: pd.DataFrame, pair: tuple, ax: plt.axes) -> None:
    ax.plot(df['Year'], df[pair[0]])
    ax.plot(df['Year'], df[pair[1]])
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("{}".format("Value"), fontsize=12)

def make_dual_plot_mixed(df: pd.DataFrame, pair: tuple, ax: plt.axes) -> None:
    ax.plot(df['Year'], df[pair[0]])
    ax2 = ax.twinx()
    ax2.plot(df['Year'], df[pair[1]], color="C1")
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel(pair[0].capitalize(), fontsize=14)
    if pair[1] == 'loudness':
        ax2.set_ylabel("Loudness (dB)", fontsize=14)
    elif pair[1] == 'tempo':
        ax2.set_ylabel("Tempo (bpm)", fontsize=14)
    else:
        ax2.set_ylabel("{}".format("Value"), fontsize=14)

def make_scatter(df: pd.DataFrame, pair: tuple, ax: plt.axes) -> None:
    ax.scatter(df[pair[0]], df[pair[1]])
    ax.set_xlabel(pair[0].capitalize(), fontsize=14)
    ax.set_ylabel(pair[1].capitalize(), fontsize=14)
    r2 = stats.pearsonr(df[pair[0]], df[pair[1]])[0]
    print("\nR^2 of " + pair[0] + " and " + pair[1] + " is " + str(r2))
