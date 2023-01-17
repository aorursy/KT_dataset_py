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
from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules



pd.options.display.max_columns=100
data = pd.read_csv('../input/basket-optimisation/Market_Basket_Optimisation.csv')
data.head()
def encode_units(x):

    if x != 0:

        return 1

    else:

        return 0
hot_encoded_df=data.fillna(0)

hot_encoded_df = hot_encoded_df.applymap(encode_units)

hot_encoded_df
frequent_itemsets = apriori(hot_encoded_df, min_support=0.02, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)

print("频繁项集：", frequent_itemsets)

print("关联规则：", rules[ (rules['lift'] >= 1) & (rules['confidence'] >= 0.5) ])
from wordcloud import WordCloud

from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt

from PIL import Image
# 生成词云

def create_word_cloud(f):

	print('根据词频，开始生成词云!')

	cut_text = word_tokenize(f)

	#print(cut_text)

	cut_text = " ".join(cut_text)

	wc = WordCloud(

		max_words=10,

		width=4000,

		height=2400,

    )

	wordcloud = wc.generate(cut_text)

	# 写词云图片

	wordcloud.to_file("wordcloud.jpg")

	# 显示词云文件

	plt.imshow(wordcloud)

	plt.axis("off")

	plt.show()
s = pd.read_csv('../input/basket-optimisation/Market_Basket_Optimisation.csv')

s.fillna('None',inplace=True)

col = list(s)

all_word = ''

for c in col:

    for w in s[c]:

        if w != 'None':

            all_word += w
# 生成词云

create_word_cloud(all_word)