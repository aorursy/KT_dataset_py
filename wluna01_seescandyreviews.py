# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
reviews_filepath = "../input/nuts_reviews.csv"

reviews_data = pd.read_csv(reviews_filepath, index_col=0, encoding="latin-1")
reviews_data.head()
reviews_data.reset_index()
reviews_data['rating'].value_counts()
high_rating_data = reviews_data[reviews_data['rating'] == 5]

low_rating_data = reviews_data[reviews_data['rating'] != 5]
high_rating_words = pd.DataFrame(high_rating_data['word'].value_counts())

low_rating_words = pd.DataFrame(low_rating_data['word'].value_counts())

high_rating_words = high_rating_words[high_rating_words['word'] > 1]

low_rating_words = low_rating_words[low_rating_words['word'] > 1]
high_rating_words.head()
high_rating_words = high_rating_words.reset_index()

high_rating_words = high_rating_words.reset_index()

high_rating_words = high_rating_words.rename(columns = {'level_0':'high_rank', 'index':'word','word':'high_frequency'})



low_rating_words = low_rating_words.reset_index()

low_rating_words = low_rating_words.reset_index()

low_rating_words = low_rating_words.rename(columns = {'level_0':'low_rank', 'index':'word','word':'low_frequency'})

#determine the discrepancy between the lengths of the two lists in order to scale the rank values later on

(high_rating_words['high_rank'].nunique())/(low_rating_words['low_rank'].nunique())
final_table = pd.merge(high_rating_words, low_rating_words,

         how='outer', on='word')
final_table = final_table.drop(columns=['high_frequency','low_frequency'])
final_table.head(10)
#replace all null values with 3094, to treat it as if it

final_table = final_table.fillna(1112.0)
final_table.sample(20)
final_table['rank_difference'] = final_table.apply(lambda row: row.high_rank + row.low_rank, axis=1)
final_table.sample(20)
high_rating_words.to_csv('high_rating_words.csv',index=False)

low_rating_words.to_csv('low_rating_words.csv',index=False)