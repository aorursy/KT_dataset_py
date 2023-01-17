

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output



%matplotlib inline

df = pd.read_csv("../input/complete.csv")

df

#df.vote_average.plot.hist(); # semikoolon väldib ebaolulise tekstilise kribukrabu kuvamist
