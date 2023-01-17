# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
# Read the csv file into DataFrame
df = pd.read_csv('../input/bhagavad-gita.csv')
# df = pd.read_csv('bhagavad-gita.csv')
# Drop the first cloumn
df.drop(df.columns[[0]], axis=1, inplace=True)
# Check first 5 rows of the DataFrame
df.head()
# Checking the data type of every column.
df.dtypes
# Describe the DataFrame to get some stats.
df.describe()
# Finding common pattern.

# Storing the title numbers to count the number
# of verses in a chapter
title_no = df['title'].astype(int)
list_counts = title_no.value_counts()
# Checking the type of the list_count variable
type(list_counts)
# To find the total number of verses
list_counts.sum()
# Describing the Series to look at some stats
list_counts.describe()
# Setting the dimensions of the figure
plt.figure(figsize=(15,5), frameon=False)
plt.tick_params(labelsize=11, length=6, width=2)

# Passing the data to plot
sns.countplot(list_counts)
plt.xlabel("Number of verses", fontsize=18)
plt.ylabel("Counts (Chapter(s))", fontsize=18)

# Displaying the plot
plt.show()
# Setting plot dimenstions
plt.figure(figsize=(15,5), frameon=False)
plt.tick_params(labelsize=11, colors='k', length=6, width=2)

sns.set(style="darkgrid")

# Passing the data to plot
sns.countplot(title_no, color='c')
plt.xlabel("Chapter", fontsize=18)
plt.ylabel("Number of verses", fontsize=18)

# Displaying the plot
plt.show()
stopwords = set(STOPWORDS)
# Collecting the words for wordcloud
data = df['verse_text']
# Inspecting the collected words
data.head()
fig = plt.figure(figsize=(20,10), facecolor='k')
wordcloud = WordCloud(width=1300, height=600, stopwords=stopwords).generate(str(data))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
