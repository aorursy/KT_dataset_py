import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



from subprocess import check_output
city = pd.read_csv('../input/movehub-city-rankings/cities.csv')

movehubcostofliving = pd.read_csv('../input/movehub-city-rankings/movehubcostofliving.csv')

movehubqualityoflife = pd.read_csv('../input/movehub-city-rankings/movehubqualityoflife.csv')

city.to_csv('cities.csv', index=False)

movehubcostofliving.to_csv('movehubcostofliving.csv', index=False)

movehubqualityoflife.to_csv('movehubqualityoflife.csv', index=False)