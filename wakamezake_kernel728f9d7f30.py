!pip install billboard.py
import billboard

import numpy as np

import pandas as pd

from pathlib import Path
%ls ../input/top-spotify-songs-from-20102019-by-year/
dataset_path = Path("../input/top-spotify-songs-from-20102019-by-year")

spotify_top_10 = pd.read_csv(dataset_path / "top10s.csv", encoding='ISO-8859-1')

chart = billboard.ChartData('hot-100', date="2019-12-01")
spotify_top_10.head()
spotify_top_10[spotify_top_10["year"] == 2019]
chart
print(chart)
for idx, song in enumerate(chart, 1):

    if song.title == "Memories":

        print(idx)

        print(song.title)

        print(song.artist)

        print(song.weeks)