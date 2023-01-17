# data processing
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# graphing
import plotly.express as px
import seaborn as sns # used for plotting correlation graph
import matplotlib.pyplot as plt

from textblob import TextBlob
df = pd.read_csv('/kaggle/input/urban-dictionary-words-dataset/urbandict-word-defs.csv', error_bad_lines=False)
df.head()
df.shape
df.isna().sum()
df = df.dropna()
df.isna().sum()
ds = df[['word', 'up_votes', 'down_votes', 'definition']].copy()
ds.loc['up_votes'] = pd.to_numeric(ds['up_votes'], errors='coerce')
ds.loc['down_votes'] = pd.to_numeric(ds['down_votes'], errors='coerce')
ds = ds.dropna()
ds.head()
ds[ds.down_votes.isin([0, 0.0])] = 1
ds[ds.up_votes.isin([0, 0.0])] = 1
ds = ds[ds.down_votes > 0]
ds = ds[ds.up_votes > 0]
ds['ratio_of_votes'] = ds.up_votes.div(ds.down_votes)
ds['total_votes'] = ds.up_votes.add(ds.down_votes)
ds.sort_values(by=['ratio_of_votes'], ascending=False).tail(10)
ds = ds.sort_values(['ratio_of_votes'])
fig = px.bar(
    ds.tail(40), 
    x="ratio_of_votes", 
    y="word", 
    text='definition',
    orientation='h', 
    title='Best ratio of votes', 
    width=2000, 
    height=800
)
fig.update_traces(textposition='outside')
fig.show()
ds = ds.sort_values(['ratio_of_votes'], ascending=False)
fig = px.bar(
    ds.tail(40), 
    x="ratio_of_votes", 
    y="word", 
    text='definition',
    orientation='h', 
    title='Worst ratio of votes', 
    width=2000, 
    height=800
)
fig.update_traces(textposition='outside')
fig.show()
ds1 = df['author'].value_counts().reset_index()
ds1.columns = ['author', 'count']
ds1 = ds1.sort_values(['count'])
fig = px.bar(
    ds1.tail(40), 
    x="count", 
    y="author", 
    orientation='h', 
    title='Top 40 authors by number of publications (authors are just hashes of the name :( )', 
    width=800, 
    height=800
)
fig.show()
ds['definition'] = ds['definition'].apply(str)
ds.head()
polarity = []
subjectivity = []
for i in range(ds.shape[0]):
    blob = TextBlob(ds['definition'].values[i])
    sentiment = blob.sentiment
    polarity.append(sentiment[0])
    subjectivity.append(sentiment[1])
ds['polarity'] = polarity
ds['subjectivity'] = subjectivity
f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(16, 8))
plt.style.use('seaborn-pastel')
ax1.boxplot(ds.polarity)
ax1.set_title('Polarity')
ax2.set_title('Subjectivity')
ax2.boxplot(ds.subjectivity);
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(16, 20))
ax1.hist(ds.polarity, bins=20, density=True, color='skyblue')
ax1.set_title("Histogram of Polarity")
ax2.hist(ds.polarity[ds.ratio_of_votes >15], bins=20, color='lightgreen', density=True)
ax2.set_title("Polarity with ratio of votes > 15")
ax3.hist(ds.polarity[ds.ratio_of_votes > 50], bins=20, color='lightpink', density=True)
ax3.set_title("Polarity with ratio of votes > 50");
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(16, 20))
ax1.hist(ds.subjectivity, bins=20, density=True, color='skyblue')
ax1.set_title("Subjectivity")
ax2.hist(ds.subjectivity[ds.ratio_of_votes >15], bins=20, color='lightgreen', density=True)
ax2.set_title("Subjectivity with ratio of votes > 15")
ax3.hist(ds.subjectivity[ds.ratio_of_votes > 50], bins=20, color='lightpink', density=True)
ax3.set_title("Subjectivity with ratio of votes > 50");
# words, definitions that obtained completely positive sentiment values
print(ds.loc[ds.polarity == 1, 'definition'].head(10))
print(ds.loc[ds.polarity == 1, 'word'].head(10))
plt.rcParams['figure.figsize'] = (16, 8)
plt.style.use('fast')

sns.set(style="whitegrid")
corr = ds.corr()
sns.heatmap(corr,annot=True,cmap="coolwarm")