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
train = pd.read_csv('../input/train.csv')

songs = pd.read_csv('../input/songs.csv')
songs.head()
train.head()
listen_log = train[['msno','song_id','target']].merge(songs[['song_id','artist_name','genre_ids']],on='song_id')

listen_log.head()
print('20 Most repeated artists:')

listen_log[listen_log.target==1].groupby('artist_name').size().nlargest(20).reset_index(name='count')
print('20 Most skipped artists:')

listen_log[listen_log.target==0].groupby('artist_name').size().nlargest(20).reset_index(name='count')