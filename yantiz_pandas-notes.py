# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])
pd.Series([1, 2, 3, 4, 5])
pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")
wine_reviews.shape
wine_reviews.head()
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
reviews
reviews.country
#reviews['country']
reviews['country'][0]
# To select the first row of data in a DataFrame, we may use the following:
reviews.iloc[0]
# To get a column with iloc, we can do the following:
reviews.iloc[:, 0]
# It's also possible to pass a list:
reviews.iloc[[0, 1, 2], 0]
# For example, to get the first entry in reviews, we would now do the following:
reviews.loc[0, 'country']
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]
reviews.set_index("title")
reviews.country == 'Italy'
reviews.loc[reviews.country == 'Italy']
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]
#reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]
reviews.loc[reviews.country.isin(['Italy', 'France'])]
reviews.loc[reviews.price.notnull()]
reviews['critic'] = 'everyone'
reviews['critic']
reviews['index_backwards'] = range(len(reviews), 0, -1)
reviews['index_backwards']
reviews.info()
reviews.points.describe()
reviews.taster_name.describe()
reviews.points.mean()
reviews.taster_name.unique()
reviews.taster_name.value_counts()
review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean)
def remean_points(row):
    row.points = row.points - review_points_mean
    return row

reviews.apply(remean_points, axis='columns')
review_points_mean = reviews.points.mean()
reviews.points - review_points_mean
reviews.country + " - " + reviews.region_1
reviews.groupby('points').points.count()
# size() also counts NaN.
reviews.groupby('points').size()
reviews.groupby('points').price.min()
reviews.groupby('winery').apply(lambda df: df.title.iloc[0])
reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])
reviews.groupby(['country']).price.agg([len, min, max])
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
countries_reviewed
mi = countries_reviewed.index
type(mi)
countries_reviewed.reset_index()
countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='len')
#countries_reviewed.sort_values(by='len', ascending=False)
countries_reviewed.sort_index()
countries_reviewed.sort_values(by=['country', 'len'])
reviews.price.dtype
reviews.dtypes
reviews.points.astype('float64')
reviews.index.dtype
reviews[pd.isnull(reviews.country)]
#reviews[reviews.country.isnull()]
reviews.region_2.fillna("Unknown")
#reviews.region_2.fillna(method='ffill')
#reviews.region_2.fillna(method='bfill')
reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")
reviews.rename(columns={'points': 'score'})
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})
reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')
canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")

pd.concat([canadian_youtube, british_youtube])
left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

left.join(right, lsuffix='_CAN', rsuffix='_UK')