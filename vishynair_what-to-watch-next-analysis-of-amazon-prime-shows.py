with plt.xkcd():
    # Based on "The Data So Far" from XKCD by Randall Munroe
    # https://xkcd.com/373/

    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax.bar([0, 1], [0, 100], 0.25)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['KAGGLE', 'BINGE WATCHING'])
    ax.set_xlim([-0.5, 1.5])
    ax.set_yticks([])
    ax.set_ylim([0, 110])

    ax.set_title("Why I have Dark Circles")

    fig.text(
        0.5, -0.05,
        'If you said Binge watch TV and TACOS, I AM IN!',
        ha='center')

plt.show()
#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')
#To display the full text column instead of truncating one
pd.set_option('display.max_colwidth', -1)
#Importing the dataset
prime = pd.read_csv("/kaggle/input/amazon-prime-tv-shows/Prime TV Shows Data set.csv",encoding="iso-8859-1")
#Head of the dataset
prime.head()
#Dimension of the Dataset
prime.shape
#Info the dataset
prime.info()
#Just an initial statistic
prime.describe()
#Just trying to plot the above seen initial stats
prime.describe().plot(kind='area',fontsize=10,figsize=(20,8),colormap='rainbow')
plt.title("Initial Stats for the Dataset",fontweight='bold')
plt.ylabel("Value")
plt.show()
#Just a step of cleaning
prime.drop(['S.no.'],axis=1,inplace=True)
prime.head()
#Finding out the show with the maximum no of words
prime['Name of the show'].loc[prime['Name of the show'].map(lambda x: len(x.split())) == 
                              prime['Name of the show'].map(lambda x: len(x.split())).max()]
#Finding the show whose title is having maximum characters
prime['Name of the show'].loc[prime['Name of the show'].map(lambda x: len(x)) ==
                              prime['Name of the show'].map(lambda x: len(x)).max()]
prime.loc[prime['Year of release']==prime['Year of release'].min()]
prime_latest = prime.loc[prime['Year of release']==prime['Year of release'].max()]
prime_latest.style.background_gradient(cmap='Greens')
sns.set_style('whitegrid')
plt.figure(figsize=(20,5))
sns.countplot(x='Genre',data=prime_latest,color="#d0d0d0",edgecolor='black')
plt.xticks(rotation=45)
plt.title("How many option in each GENRE for this year releases",fontweight='bold')
plt.suptitle("* Just considering films released in 2020",color='grey')
plt.show()
prime_seasons = prime.sort_values(['No of seasons available'],axis=0,ascending=False)[:5]
prime_seasons.style.background_gradient(cmap='Blues')
#Just a count check rather than plotting
from collections import Counter
Counter(prime['Language']).keys()
Counter(prime['Language'])
Counter(prime['Language']).keys()
sns.set_style("whitegrid")
plt.figure(figsize=(10,10))
plt.title("Which all Genre and how many of them?",fontweight='bold')
sns.countplot(y='Genre',data=prime,color='skyblue',edgecolor='black')
plt.show()
prime.sort_values(['IMDb rating'],axis=0,ascending=False)[:5]
plt.figure(figsize=(8,8))
plt.title("TV SHOWS FOR WHICH AGE GROUP",fontweight='bold')
ax = sns.countplot(x='Age of viewers',data=prime,palette='Wistia',edgecolor='black')
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
sizes = Counter(prime['Age of viewers']).values()
labels = Counter(prime['Age of viewers']).keys()

explode = (0.1, 0.1, 0.1, 0.1, 0.1) 
  
# Creating color parameters 
colors = ( "orange", "lightcoral", 
          "lightskyblue", "yellowgreen", "beige") 
# Plot
plt.figure(figsize=(8,8))
plt.title("Viewership Age - All vs Rest",fontweight='bold')
plt.pie(sizes, explode=explode, labels=labels, colors=colors,shadow=True,autopct='%1.1f%%',startangle=140)
plt.axis('equal')
plt.show()
plt.figure(figsize=(15,5))
plt.title("Languages vs No. of Seasons",fontweight='bold')
sns.boxplot(y=prime['No of seasons available'],x=prime['Language'])
plt.show()
plt.figure(figsize=(10,5))
plt.title("Does overall IMDB fall with more number of seasons?",fontweight='bold')
plt.xlabel("Number of Seasons")
plt.ylabel("IMDB Rating")
plt.scatter(prime['No of seasons available'],prime['IMDb rating'])
plt.grid(linestyle='dotted')
plt.show()
#Trying jointplot fo these two features to understand the underlying relationship more clearly
g = (sns.jointplot("No of seasons available", "IMDb rating",data=prime, color="k").plot_joint(sns.kdeplot, zorder=0, n_levels=6))
fig, ax = plt.subplots(1,1,figsize=(12,6))
ax.scatter(x='Age of viewers',y='IMDb rating',data=prime[prime['Genre']=='Drama'],color='skyblue',alpha=0.5,
          label='Drama',s=70)
ax.scatter(x='Age of viewers',y='IMDb rating',data=prime[prime['Genre']=='Kids'],color='salmon',alpha=0.5,
          label='Kids',s=70)
ax.scatter(x='Age of viewers',y='IMDb rating',data=prime[prime['Genre']=='Comedy'],color='#383838',alpha=0.5,
          label='Comedy',s=70)
ax.set_title("Age of Viewers vs IMDB vs Genre",fontweight='bold')

ax.legend(bbox_to_anchor=(1.2, 1))

# upper & right border remove 
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()