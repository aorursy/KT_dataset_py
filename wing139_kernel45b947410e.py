# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_2 = pd.read_excel("/kaggle/input/online-retail-data-set-from-uci-ml-repo/Online Retail.xlsx")

#df_2.head(15)

# create a figure and axis

fig, ax = plt.subplots()



def convert_excel_time(excel_time):

    return pd.to_datetime('1900-01-01') + pd.to_timedelta(excel_time,'D')



# scatter the sepal_length against the sepal_width

ax.scatter(df_2['InvoiceDate'], df_2['UnitPrice'])

# set a title and labels

ax.set_title('Dataset')

ax.set_xlabel('Date')

ax.set_ylabel('UnitPrice')
df_2 = pd.read_excel("/kaggle/input/online-retail-data-set-from-uci-ml-repo/Online Retail.xlsx")

df_2[["InvoiceNo", "StockCode","Description"]].head()

#InvoiceNo	StockCode	Description	Quantity	InvoiceDate	UnitPrice	CustomerID	Country



text = df_2.Description[0]



# Create and generate a word cloud image:

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(df_2['Description'])



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()