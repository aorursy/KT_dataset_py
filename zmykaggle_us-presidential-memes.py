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
%matplotlib inline
def get_data(first, last):

    """ get data for person named first, last"""

    df1 = pd.read_csv('../input/{}.csv'.format(first), low_memory=False,

                 header=0, sep=',', encoding = "ISO-8859-1", parse_dates = ['timestamp'],

                     usecols=['timestamp','network','author','likes'])

    df2 = pd.read_csv('../input/{}.csv'.format(last), low_memory=False,

                 header=0, sep=',', encoding = "ISO-8859-1", parse_dates = ['timestamp'],

                     usecols=['timestamp','network','author','likes'])

    return pd.concat([df1, df2]).drop_duplicates()
donald = get_data("Donald", "Trump")

hillary = get_data("Hillary", "Clinton")

donald.info()

print("-"*80)

hillary.info()
donald.head()
hillary['year'] = hillary['timestamp'].apply(lambda t:t.year)

hillary['month'] = hillary['timestamp'].apply(lambda t:t.month)
hillary_agg = hillary.groupby([hillary['year'], hillary['month']])['likes'].sum()
donald['year'] = donald['timestamp'].apply(lambda t:t.year)

donald['month'] = donald['timestamp'].apply(lambda t:t.month)

donald_agg = donald.groupby([donald['year'], donald['month']])['likes'].sum()
frames = [hillary_agg, donald_agg]

data = pd.concat(frames, axis = 1)
data.columns = ['Hillary', 'Trump']

data.plot(figsize = [9,6])
donald_network=donald.groupby([donald['network']])['likes'].sum()
hillary_network=hillary.groupby([hillary['network']])['likes'].sum()
network_data = pd.concat([donald_network, hillary_network], axis=1)

network_data.columns=['Trump','Hillary']
network_data.head()
network_data.plot(kind='bar')
donald_by_author = donald.groupby(['author']).likes.sum().sort_values(ascending=False)
donald_by_author.head(10)
# Aggregate the rest into 'Other'

d = pd.Series(donald_by_author[10:].sum(), index=["Other"])

d = pd.concat([donald_by_author[:10], d], axis=0)
d.head(11).plot(kind='pie', autopct='%1.1f%%', startangle=270, fontsize=8, 

                title="Trump 'likes' distribution",)
hillary_by_author = hillary.groupby('author').likes.sum().sort_values(ascending=False)
hillary_by_author.head(10)
h = pd.Series(hillary_by_author[10:].sum(), index=["Other"])

h = pd.concat([hillary_by_author[:10], h], axis=0)

h.head(11).plot(kind='pie', autopct='%1.1f%%', startangle=270, fontsize=8, 

                title="Hillary 'likes' distribution",)