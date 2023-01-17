# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_set = pd.read_csv('../input/movie_metadata.csv')

data_set
# Get all values for film duration

duration_set = data_set['duration']

duration_set

# Get all values for film gross

gross_set = data_set['gross']

gross_set
plt.plot(duration_set, gross_set, 'bo')

plt.ylabel('Film gross (in powers of 10)')

plt.xlabel('Film duration (in minutes)')

plt.show()
pg_set = data_set[data_set['content_rating'].isin(['PG'])]

pg_set
pg13_set = data_set[data_set['content_rating'].isin(['PG-13'])]

pg13_set
r_set = data_set[data_set['content_rating'].isin(['R'])]

r_set
pg_gross = pg_set['gross']

pg_grossIQR = pg_gross.quantile(.75) - pg_gross.quantile(.25)

pg_grossIQR
pg_gross.describe()
pg_gross.mean()
pg13_gross = pg13_set['gross']

pg13_grossIQR = pg13_gross.quantile(.75) - pg13_gross.quantile(.25)

pg13_grossIQR
pg13_gross.describe()
pg13_gross.mean()
r_gross = r_set['gross']

r_grossIQR = r_gross.quantile(.75) - r_gross.quantile(.25)

r_grossIQR
r_gross.describe()
r_gross.mean()
ratings_arr = {'IMDB Rating': data_set['imdb_score'], 'Film Gross': data_set['gross']}

ratings_df = pd.DataFrame(ratings_arr)

ratings_df.corr('pearson')
#plt.plot(duration_set, gross_set, 'bo')

plt.scatter(data_set['imdb_score'], gross_set)

plt.ylabel('Film gross (in powers of 10)')

plt.xlabel('IMDB Rating')

plt.show()