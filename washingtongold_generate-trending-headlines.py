import numpy as np

import pandas as pd

data = pd.read_csv('/kaggle/input/youtube-new/USvideos.csv')

data
data['views'].mean()
m = data[(data['category_id']==24) | (data['category_id']==23) | (data['category_id']==22)]

from sklearn.utils import shuffle

m = shuffle(shuffle(shuffle(m)))
titles = open('title.txt','w+')

for item in m['title'].head(1_000):

    titles.write(item)

    titles.write('\n')

titles.close()
print(open('title.txt','r').read())
!pip install textgenrnn
from textgenrnn import textgenrnn

textgen = textgenrnn()

textgen.train_from_file('title.txt', num_epochs=50)
textgen.generate()
textgen.generate()
textgen.generate()
textgen.generate()
textgen.generate()
m = shuffle(shuffle(shuffle(m)))

titles = open('title.txt','w+')

for item in m['title'].head(1_000):

    titles.write(item)

    titles.write('\n')

titles.close()

from textgenrnn import textgenrnn

textgen = textgenrnn()

textgen.train_from_file('title.txt', num_epochs=50)