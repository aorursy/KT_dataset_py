import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
data = '../input/ultimate-spotify-tracks-db/SpotifyFeatures.csv'
spotifydata = pd.read_csv(data)
spotifydata.head()
spotifydata.info()
spotifydata.isnull().sum()
spotifydata.describe().T
spotifydata.groupby("genre").popularity.mean().sort_values(ascending = False).head()
spotifydata.drop_duplicates(subset = ["track_id"],inplace = True)
spotifydata.groupby("genre").popularity.mean().sort_values(ascending = False).head()
topdata = spotifydata.loc[spotifydata.genre.isin(["Pop","Dance","Hip-Hop","Rap","Rock","Dance"])]
topdata
plt.figure(figsize =(15,5))

ax1 = plt.subplot(1,2,1)
sns.barplot(x = topdata["genre"], y=topdata["popularity"])
plt.xlabel("Genre",fontsize = 12)
plt.ylabel("Popularity",fontsize = 12)
plt.title("Genre vs Popularity")

ax2 = plt.subplot(1,2,2)
sns.countplot(x=topdata["genre"], data=topdata)
plt.xlabel("Genre",fontsize = 12)
plt.ylabel("No. of Artists",fontsize = 12)
plt.title("No. of Artists per Genre")
sns.barplot(x = topdata["popularity"], y=topdata["time_signature"])
plt.title("Popularity vs Time Signature")
sns.barplot(x = topdata["key"], y= topdata["popularity"])
plt.title("Key vs Popularity")
plt.figure(figsize=(30,15))
pltnum = 1

for col in ["acousticness","danceability","duration_ms","energy","instrumentalness","liveness","loudness", "speechiness", "tempo", "valence"]:
    if pltnum<=10:
        ax = plt.subplot(2,5, pltnum)
        sns.scatterplot(x =col, y="popularity", data=topdata, hue= "mode", legend = "full")
        plt.xlabel(col,fontsize = 17)
        plt.ylabel("Popularity",fontsize = 17)
        plt.legend(fontsize = 15)
    pltnum +=1
plt.suptitle("Relationship between Musical Attributes and Popularity",fontsize = 23)
plt.show()
plt.figure(figsize=(30,15))
pltnum = 1

for col in ["acousticness","danceability","duration_ms","energy","instrumentalness","liveness","loudness", "speechiness", "tempo", "valence"]:
    if pltnum<=10:
        ax = plt.subplot(2,5, pltnum)
        sns.distplot(a =topdata[col])
        plt.xlabel(col,fontsize = 17)
    pltnum +=1
plt.suptitle("Distribution of Musical Attributes",fontsize = 23)
plt.show()
plt.figure(figsize = (12, 8))

sns.heatmap(topdata.corr(), annot=True )
plt.title("Correlation between features", fontsize = 15)
print ("Total number of artists in the top 5 genres: "+str(len(topdata.artist_name.unique())))
topdata.groupby("artist_name").popularity.max().sort_values(ascending=False).head(5)
pop5 = topdata.loc[topdata.artist_name.isin(["Ariana Grande","Post Malone","Daddy Yankee","Sam Smith","Halsey"])]
pop5.groupby("genre").artist_name.unique()
fig = px.violin(pop5, x = "artist_name", y="popularity", points="all", hover_data=pop5.columns, color =pop5.artist_name)
fig.show()
plt.figure(figsize = (20, 35))
pltnum = 1

for col in ["acousticness","danceability","duration_ms","energy","instrumentalness","liveness","loudness", "speechiness", "tempo", "valence"]:
    if pltnum<=10:
        ax = plt.subplot(5,2,pltnum)
        sns.boxplot(x = pop5["artist_name"], y= pop5[col])
        plt.xlabel("Artist Name", fontsize=12)
        plt.ylabel(col,fontsize=15)
    pltnum +=1
plt.show