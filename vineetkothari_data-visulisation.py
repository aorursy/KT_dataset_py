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
df = pd.read_csv("../input/movie_metadata.csv")

#Remove duplicates if any refered from The Money Makers

df_clean = df.drop_duplicates(['movie_title'])
df.head()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_context("poster")

sns.set_style("ticks")


#getting columns index 

df.columns

#ADD PROFIT AND ROI

#profit = gross - budget

profit = (df['gross']-df['budget'])

#return_on_investment_perc

return_on_investment_perc=(profit/df['gross'])*100;

#INSERT EXTRA COLUMNS

df.insert(len(df.columns),'profit',profit)

#df.insert(len(df.columns),'percentage',return_on_investment_perc)
df_t = df.pivot_table(index="profit", values="color", aggfunc=len)

df_t.head()
plt.plot(df_t.index, df_t.values, "-bo")

plt.xlabel("Year")

plt.ylabel("Total movies")

plt.yscale("log")
def plot_vs_year(df, y="gross"):

    df_t = df.pivot_table(index="title_year", values=y, aggfunc=np.mean)

    plt.plot(df_t.index, df_t.values, "-bo")

    plt.ylabel(y)

    plt.xlabel("Year")
plot_vs_year(df, "profit")
#categorical values

ax = sns.countplot(x="language", data=df)

ax.set_yscale("log")

labels = ax.get_xticklabels()

_ = plt.setp(labels, rotation=90, fontsize=10)
df.plot_keywords.head()
df_keywords = df.plot_keywords.apply(lambda x: [] if isinstance(x, float) else x.split("|"))

df_keywords.head()
df_plots = pd.Series(sum(df_keywords.values.tolist(), []))

df_plots.head()
plot_counts = df_plots.value_counts().sort_values(ascending=False)

print("Plot counts: ", plot_counts.shape)

plot_counts.head()
plot_counts[plot_counts > 50].index
ax = sns.countplot(x=df_plots[df_plots.isin(plot_counts[plot_counts > 50].index.values)])

ax.set_yscale("log")

labels = ax.get_xticklabels()

_ = plt.setp(labels, rotation=90, fontsize=10)
#selecting all movies having profit greater than 1000

profit = df[df['profit'] > 1000]

#profit['movie_title']

#profit['profit']

#factor plot
g = sns.FacetGrid(tidy, col='movie_title', col_wrap=6, hue='movie_title')

g.map(sns.barplot, 'variable', 'rest');
