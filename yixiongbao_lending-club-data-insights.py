import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read files

df = pd.read_csv('../input/loan.csv',low_memory=False)
df.shape
df.dtypes
df['loan_amnt'].plot.hist()

plt.title('Loan Amount Distribution')

plt.ylabel('Frequency')

plt.xlabel('Amount')
df['loan_amnt'].plot.density()

plt.xlabel('Amount')

plt.title('Loan Amount Density')
df['loan_amnt'].plot.box()

plt.title('Loan Amount Boxplot')

plt.ylabel('Loan Amount')

plt.xlabel('')
df['loan_status'].value_counts()
df.groupby('loan_status')['loan_amnt'].sum().sort_values(ascending=0).plot(kind='bar')

plt.xlabel('Loan Status')

plt.ylabel('Loan Amount')

plt.title('What kind of loan status have the largest amount?')
df['purpose'].value_counts().head(n=10)
from os import path

from wordcloud import WordCloud

plt.figure(figsize=(10,7))

text = df['title'].to_json()

wc = WordCloud(ranks_only=True,prefer_horizontal = 0.6,background_color = 'white',

              max_words = 50).generate(text)

plt.imshow(wc)

plt.axis("off")
df['title'].value_counts().head(n=10)
df['grade'].value_counts().sort_index().plot(kind='bar')

plt.title('Loan Grade Volume Distribution')

plt.xlabel('Grade')

plt.ylabel('Volume of Loans')
df.groupby('grade')['loan_amnt'].sum().sort_index().plot(kind='bar')

plt.title('Loan Grade Amount Distribution')

plt.xlabel('Grade')

plt.ylabel('Amount of Loans')
df['issue_d'] = pd.to_datetime(df.issue_d)

df.groupby('issue_d')['loan_amnt'].sum().plot()

plt.title('Trends of loans amount issued')

plt.xlabel('Year')

plt.ylabel('Loan Amount')
df['issue_d'].value_counts().sort_index().plot(kind='line')

plt.xlabel('Year')

plt.ylabel('Loan Volume')

plt.title('Trends of Loan Volume')
df['issue_Y'] = df['issue_d'].dt.year

temp = df.groupby(['grade','issue_Y'],as_index=False)['id'].count()



import matplotlib.cm as cm

dpoints = np.array(temp)

fig = plt.figure(figsize=(15,7))

ax = fig.add_subplot(1,1,1)

space = 0.3

conditions = np.unique(dpoints[:,0])

categories = np.unique(dpoints[:,1])

n = len(conditions)

width = (1-space)/len(conditions)



for i,cond in enumerate(conditions):

    vals = dpoints[dpoints[:,0] == cond][:,2].astype(np.float)

    pos = [j - (1 - space) / 2. + i * width for j in range(1,len(categories)+1)]

    ax.bar(pos, vals, width = width,label=cond, 

       color=cm.Accent(float(i) / n))

    ax.set_xticklabels(['','2008','2010','2012','2014',''])

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles[::-1], labels[::-1])

    ax.set_ylabel("Loan Volume")

    ax.set_xlabel("Year")

plt.title('Loan Volume Trends by Grade')