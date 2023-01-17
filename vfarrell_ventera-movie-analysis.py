# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
movies = pd.read_csv('../input/movie_metadata.csv');

%matplotlib inline
movies.columns
ratings = movies.imdb_score.unique()

ratings.sort()

ratings
rValues = movies.corr().imdb_score

rValues.sort_values()
rSquared = rValues*rValues

rSquared.sort_values()
#movies.director_name

movies.groupby(['director_name', 'imdb_score']).mean()
directorScores = movies[['director_name', 'imdb_score']]

directorScores = directorScores.reset_index()

directorScores.columns = ['ID', 'director_name', 'imdb_score']

directorScores

directorScores.sort_values(by='director_name')

dir_score_sum = directorScores[['director_name', 'imdb_score']].groupby(['director_name']).agg(['sum']).reset_index()
directorCounts = directorScores[['director_name', 'ID']].groupby(['director_name']).agg(['count']).reset_index()
foo = pd.concat([dir_score_sum, directorCounts], axis=1)
foo.reset_index(inplace=True)
doit = foo.as_matrix()

foo2 = pd.DataFrame(doit)

foo2.columns = ['ID', 'ID2', 'director_name', 'imdb_sum','director_name2', 'movie_count']

director_data = foo2[['director_name', 'imdb_sum','movie_count']]
director_data