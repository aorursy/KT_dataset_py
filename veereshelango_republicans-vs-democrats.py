# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud,STOPWORDS



pal = sns.color_palette()

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/inaug_speeches.csv",encoding="iso-8859-1")

df = df.iloc[:,1:]

df.head()
print("Total number of speeches :",len(df))
df.isnull().any()
#Number of words in each speech

df["word_count"] = df["text"].apply(lambda x : len(x))

#Number of unique words in each speech

df["unique_word"] = df["text"].apply(lambda x : len(set(x.lower().split()) ) )

#Number of unique words ratio in each speech

df["unique_word_ratio"] = df.apply(lambda x : x["unique_word"]/x["word_count"] ,axis=1)

#Extracting year alone from the Date column

df["year"] = df["Date"].apply(lambda x : int(x.split(",")[2])  if len(x.split(","))==3 else int(x.split(",")[1]) )

df.head()
party_dict = {

    "Federalist":["George Washington","John Adams"],

    "Democratic-Republican":['Thomas Jefferson',

       'James Madison', 'James Monroe', 'John Quincy Adams'],

    "Democrat":['Andrew Jackson', 'Martin Van Buren',

                'James Knox Polk','Franklin Pierce',

                'James Buchanan','Grover Cleveland',

               'Woodrow Wilson','Franklin D. Roosevelt',

       'Harry S. Truman','John F. Kennedy',

       'Lyndon Baines Johnson','Jimmy Carter',

                'Bill Clinton', 'Barack Obama'

               ],

    "Whig":["William Henry Harrison",'Zachary Taylor'],

    "Republican":['Abraham Lincoln', 'Ulysses S. Grant',

       'Rutherford B. Hayes', 'James A. Garfield','Benjamin Harrison', 'William McKinley', 'Theodore Roosevelt',

       'William Howard Taft','Warren G. Harding',

       'Calvin Coolidge', 'Herbert Hoover',

                  'Dwight D. Eisenhower',

                  'Richard Milhous Nixon',

                  'Ronald Reagan', 'George Bush',

                  'George W. Bush','Donald J. Trump'

                 ]

    

}

def get_party(name):

    for party,names in party_dict.items():

        if name in names:

            return party

df["party"] = df["Name"].apply(lambda x : get_party(x))

g= sns.countplot(y="party",data=df)

plt.title("Number of Speeches per Political Party")
ax= sns.boxplot(df["word_count"],orient='v')

plt.title("Box plot of Word Count")
print("Average number of words per speech : ",df["word_count"].mean())

print()

print("Smallest Speech : ")

print(df.ix[df["word_count"].idxmin(axis=1)])

print()

print("Longest Speech : ")

print(df.ix[df["word_count"].idxmax(axis=1)])
wc_mean = df.groupby("party")["word_count"].mean().reset_index()

g =sns.factorplot(x="party",y="word_count",kind="bar",data=wc_mean)

g.set_xticklabels(rotation=90)
g = sns.factorplot(x="year",y="word_count",hue="party",data=df,kind="bar",size=5,aspect=2,legend_out=False)

g.set_xticklabels(rotation=90)

plt.title("Year wise Speech Length")
fact1 = []

val1 = df[(df["party"] == "Republican") & (df["year"]< 1930)]["word_count"].mean()

val2 = df[(df["party"] == "Republican") & (df["year"]> 1930)]["word_count"].mean()

val3 = df[(df["party"] == "Democrat") & (df["year"]< 1930)]["word_count"].mean()

val4 = df[(df["party"] == "Democrat") & (df["year"]> 1930)]["word_count"].mean()

fact1.append(["before 1930",val1,"Republican"])

fact1.append(["after 1930",val2,"Republican"])

fact1.append(["before 1930",val3,"Democrat"])

fact1.append(["after 1930",val4,"Democrat"])

fact1_df = pd.DataFrame(fact1,columns=["period","value","party"])

overall_avg = df["word_count"].mean()

g = sns.factorplot(x="period",y="value",col="party",data=fact1_df,kind="bar")

#plt.plot([overall_avg,overall_avg],'k--')

#plt.title("Republicans Word Count")
ax= sns.boxplot(df["unique_word"],orient='v')

plt.title("Box Plot of Unique Words")
print("Average unique words used per speech : ",df["unique_word_ratio"].mean())

print()

print("Speech with least unique words :")

print(df.ix[df["unique_word_ratio"].idxmin(axis=1)])

print()

print("Speech with most unique words :")

print(df.ix[df["unique_word_ratio"].idxmax(axis=1)])
uni_mean = df.groupby("party")["unique_word_ratio"].mean().reset_index()

g =sns.factorplot(x="party",y="unique_word_ratio",kind="bar",data=uni_mean)

g.set_xticklabels(rotation=90)

plt.title("Unique Word Ratio per Party")
g = sns.factorplot(x="year",y="unique_word_ratio",hue="party",data=df,kind="bar",size=5,aspect=2,legend_out=False)

g.set_xticklabels(rotation=90)

plt.title("Year wise Unique Word Ratio")