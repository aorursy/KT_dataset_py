!pip install apyori
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



from pandas_profiling import ProfileReport



import time

from apyori import apriori as apriori1



from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import apriori as apriori2

from mlxtend.frequent_patterns import association_rules

from mlxtend.frequent_patterns import fpgrowth



from wordcloud import WordCloud

import matplotlib.pyplot as plt

from PIL import Image
data = pd.read_csv('/kaggle/input/basket-optimisation/Market_Basket_Optimisation.csv')
data_profile_report = ProfileReport(data, 'Data Set Profiling Report', html={'style':{'full_width':True}})
data_profile_report.to_notebook_iframe()
data.fillna(0, inplace=True)
def get_apriori_statistics(rules):

    df = pd.DataFrame(list(rules))

    first_items = []

    second_items = []

    support = []

    confidence = []

    lift = []

    for i in range(df.shape[0]):

        for idx, order_stat in enumerate(df['ordered_statistics'][i]):

            if idx == 0:

                sup = df['support'][i]

                continue

            first_items.append(str(list(order_stat[0])).lstrip("['").rstrip("']"))

            second_items.append(str(list(order_stat[1])).lstrip("['").rstrip("']"))

            support.append(sup)

            confidence.append(order_stat[2])

            lift.append(order_stat[3])

    fitems = pd.DataFrame(first_items, columns=['item1'])

    sitems = pd.DataFrame(second_items, columns=['item2'])

    dsupport = pd.DataFrame(support, columns=['support'])

    dconfidence = pd.DataFrame(confidence, columns=['confidance'])

    dlift = pd.DataFrame(lift, columns=['lift'])

    return pd.concat([fitems, sitems, dsupport, dconfidence, dlift], axis=1)
def get_transactions(df):

    transactions = []

    for i in range(0, len(df)):

        transactions.append([df.iloc[i,j] for j in range(0, df.shape[1]) if df.iloc[i,j] != 0])

    return transactions
now = time.time()

rules = apriori1(get_transactions(data), min_support=0.003, min_confidance=0.2, min_left=3)

print(time.time() - now)
get_apriori_statistics(rules)
te = TransactionEncoder()

te_ary = te.fit_transform(get_transactions(data))

df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori2(df, min_support=0.003, use_colnames=True)
association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
association_rules(frequent_itemsets, metric="lift", min_threshold=3)
now = time.time()

frequent_itemsets = fpgrowth(df, min_support=0.003, use_colnames=True)

print(time.time() - now)
association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
def get_word_cloud(df, top=10):

    sentences = list(df.values)

    words = []

    for sentence in sentences:

        for word in sentence:

            if word != 0:

                words.append(word)

    text = ' '.join(words)

    wc = WordCloud(max_words=top, width=4000, height=2400)

    wordcloud = wc.generate(text)

    plt.imshow(wordcloud)

    return wordcloud
get_word_cloud(data, top=10)