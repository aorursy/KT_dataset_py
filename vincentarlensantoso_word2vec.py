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
import multiprocessing
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

cores = multiprocessing.cpu_count() # Count the number of cores in a computer

# google_model = KeyedVectors.load_word2vec_format('/kaggle/input/gensim-word-vectors/word2vec-google-news-300/word2vec-google-news-300', binary=True)

data = np.fromfile('/kaggle/input/gensim-word-vectors/word2vec-google-news-300/word2vec-google-news-300')

df = pd.DataFrame(data)
cores
w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)
df_clean = df.dropna().drop_duplicates()

w2v_model.build_vocab(df_clean)
w2v_model.wv.most_similar(positive=["apple"])
google_model.most_similar("apple")
google_model.most_similar("dog")
google_model.most_similar("marriage")
google_model.most_similar("football")
google_model.most_similar("cat")
google_model.most_similar("patient")
google_model.most_similar("united")
google_model.most_similar("green")