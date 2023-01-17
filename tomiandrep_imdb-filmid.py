import numpy as np

import pandas as pd 



%matplotlib inline

pd.set_option('display.max_rows', 20)
sisend = pd.read_csv("../input/IMDB-Movie-Data.csv")
filmid_aastas = sisend["Year"].value_counts();

sisend.Year.plot.hist(bins=len(filmid_aastas), grid=True, rwidth=0.95);
#filmid_2016 = sisend[sisend["Year"]==2016]

filmid_aastas = sisend;

filmid_aastas.plot.scatter("Rating", "Revenue (Millions)", alpha=0.4);
pd.set_option('display.max_rows', 100)

režissööride_tulu = sisend.groupby("Director")["Revenue (Millions)"].mean()

režissööride_tulu = režissööride_tulu.sort_values(ascending=False).head(100)

režissööride_tulu