import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.colors import ListedColormap
df=pd.read_csv('../input/top50spotify2019/top50.csv',engine="python")

df.head(1)
df=df.drop(['Unnamed: 0'],axis=1)

df.head(1)
df.describe()
df.info()
df.isnull().sum()
sns.set(font_scale=2)

plt.figure(figsize=(15, 20))

sns.countplot(y='Artist.Name', data=df)
sns.set(font_scale=2)

plt.figure(figsize=(15, 20))

sns.countplot(y='Genre', data=df)
sns.set(font_scale=2)

plt.figure(figsize=(60,8))

sns.swarmplot(x="Genre", y="Energy", data=df)

plt.show()
sns.set(font_scale=2)

sns.factorplot( "Loudness..dB..","Artist.Name", data=df, kind="bar",palette="muted", legend=False, size=20)

plt.show()
# Define a variable N

N = 50

# Construct the colormap

current_palette = sns.color_palette("muted", n_colors=5)

cmap = ListedColormap(sns.color_palette(current_palette).as_hex())

colors = np.random.randint(0,5,N)

# Create a scatter plot

plt.figure(figsize=(20,8))

plt.scatter(df['Energy'],df['Danceability'], c=colors, cmap=cmap)

# Add a color bar

plt.colorbar()

# Show the plot

plt.show()
df.head(1)
f,ax=plt.subplots(2,2,figsize=(25,15))

sns.violinplot(x="Liveness",ax=ax[0][0],data=df, palette="muted")

sns.violinplot(x="Valence.",data=df,ax=ax[0][1], palette="muted")

sns.violinplot(x="Acousticness..",data=df,ax=ax[1][0], palette="muted")

sns.violinplot(x="Speechiness.",data=df,ax=ax[1][1], palette="muted")

df.plot(kind= 'box' , subplots=True, layout=(5,2), sharex=False, sharey=False, figsize=(12,20)) 
df.hist (bins=10,figsize=(20,20))

plt.show ()
from wordcloud import WordCloud 

df1=df['Track.Name'].to_string()

text = df1

wordcloud = WordCloud().generate(text)

f,ax=plt.subplots(1,1,figsize=(25,5))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# Prepare Data

df1=df.head(100)

df2= df1.groupby('Genre').size()

# Make the plot with pandas

df2.plot(kind='pie', subplots=True, figsize=(8, 8))

plt.title("Pie Chart of Various Category of Difficulty")

plt.ylabel("")

plt.show()
f,ax=plt.subplots(1,1,figsize=(25,6))

sns.kdeplot(df.loc[(df['Genre']=='canadian pop'), 'Beats.Per.Minute'], color='b', shade=True, Label='Canadian Pop')

sns.kdeplot(df.loc[(df['Genre']=='reggaeton flow'), 'Beats.Per.Minute'], color='g', shade=True, Label='Reggaeton')

sns.kdeplot(df.loc[(df['Genre']=='dance pop'), 'Beats.Per.Minute'], color='r', shade=True, Label='Dance Pop')

sns.kdeplot(df.loc[(df['Genre']=='pop'), 'Beats.Per.Minute'], color='y', shade=True, Label='Pop')

plt.xlabel('Beats Per Seconds') 

plt.ylabel('Probability Density') 
sns.set(font_scale=1.5)

plt.figure(figsize=(20,8))

corr = (df.corr())

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values,cmap="YlGnBu",annot=True,linewidths=.5, fmt=".2f")

plt.title("Pearson Correlation of all Elements")
x=df1['Valence.']

y=df1['Energy']

N = 50

colors = np.random.rand(N)

area = (25 * np.random.rand(N))**2

df3= pd.DataFrame({'X': x,'Y': y,'Colors': colors,"bubble_size":area})
plt.scatter('X', 'Y', s='bubble_size', c='Colors', alpha=0.5, data=df3)

plt.xlabel("X", size=16)

plt.ylabel("y", size=16)

plt.title("Bubble Plot with Matplotlib", size=18)
plt.style.use('seaborn')

x  = [(i+1) for i in range(10)]

y1 = df1['Danceability'][1:11]

y2 = df1['Loudness..dB..'][1:11]

y3 = df1['Liveness'][1:11] 

plt.plot(x, y1, label="radius_mean", color = 'B')

plt.plot(x, y2, label="Loudness_dB", color = 'R')

plt.plot(x, y3, label="Liveness", color = 'C')

plt.plot()

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Line Graph Example")

plt.legend()

plt.show()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Artist.Name']=le.fit_transform(df['Artist.Name'])

df['Genre']=le.fit_transform(df['Genre'])
X = df.drop(['Track.Name','Popularity'],axis=1)

y= df['Popularity']
from sklearn.preprocessing import StandardScaler 

sc_X = StandardScaler() 

X = sc_X.fit_transform(X) 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state =0)
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

model = lm.fit(X,y)

print(f'alpha = {model.intercept_}')

print(f'betas = {model.coef_}')
model.predict(X)
model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

sns.distplot(y_test - predictions, axlabel="Prediction")

plt.show()