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
movie_df = pd.read_csv(dirname + '/movies_metadata.csv')

movie_df.set_index('id')

actor_df = pd.read_csv(dirname + '/credits.csv')

actor_df.set_index('id')
merged_df = pd.concat([movie_df, actor_df], axis=1)

merged_df.head(5)
merged_df.to_csv('movies_and_credis_merged.csv')
from IPython.display import HTML



def create_download_link(title = "Download CSV file", filename = "movies_and_credis_merged.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe which was saved with .to_csv method

create_download_link(filename='movies_and_credis_merged.csv')