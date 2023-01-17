'''Ignore deprecation and future, and user warnings.'''

import warnings as wrn

wrn.filterwarnings('ignore', category = DeprecationWarning) 

wrn.filterwarnings('ignore', category = FutureWarning) 

wrn.filterwarnings('ignore', category = UserWarning) 



'''Import basic modules.'''

import pandas as pd

import numpy as np

from scipy import stats



'''Customize visualization

Seaborn and matplotlib visualization.'''

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

%matplotlib inline



'''Special Visualization'''

from wordcloud import WordCloud 

import missingno as msno



'''Plotly visualization .'''

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

py.init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook



import cufflinks as cf #importing plotly and cufflinks in offline mode  

import plotly.offline  

cf.go_offline()  

cf.set_config_file(offline=False, world_readable=True)



'''Display markdown formatted output like bold, italic bold etc.'''

from IPython.display import Markdown

def bold(string):

    display(Markdown(string))
'''Reading the data from csv files'''

data = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv')

display(data.head(3))

print('Dimension of data:', data.shape)
'''Droping unwanted variable'''

data.drop(['URL', 'ID'], axis = 1, inplace = True)
'''Variable Description'''

def description(df):

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.iloc[0].values

    summary['Second Value'] = df.iloc[1].values

    summary['Third Value'] = df.iloc[2].values

    return summary
bold('**Variable Description of Data:**')

description(data)
'''Visualization of missing variable'''

msno.matrix(data)

plt.show()
%%time

import requests

from PIL import Image

from io import BytesIO



fig, ax = plt.subplots(10,10, figsize=(12,12))



for i in range(100):

    r = requests.get(data['Icon URL'][i])

    im = Image.open(BytesIO(r.content))

    ax[i//10][i%10].imshow(im)

    ax[i//10][i%10].axis('off')

plt.show()
bold('**THE MOST FREQUENT RATING IS 4.5 MEASURED**')

plt.rcParams['figure.figsize'] = (18, 10)

ax = sns.countplot(data = data, x ='Average User Rating', palette = 'gray', alpha = 0.7, linewidth=4, edgecolor= 'black')

ax.set_ylabel('Count', fontsize = 20)

ax.set_xlabel('Average User Rating', fontsize = 20)

plt.show()
bold('**USER RATING COUNT IS HIGHLY POSITIVE SKEWED**')

plt.rcParams['figure.figsize'] = (18, 10)

ax = sns.kdeplot(data['User Rating Count'], shade = True, linewidth = 5, color = 'k')

ax.set_ylabel('Count', fontsize = 20)

ax.set_xlabel('User Rating Count', fontsize = 20)

plt.show()
bold('**MOST OF THE APPS PRICES BETWEEN 0 TO 10 DOLLARS**')

plt.rcParams['figure.figsize'] = (18, 10)

ax = sns.kdeplot(data['Price'], shade = True, linewidth = 5, color = 'm')

ax.set_ylabel('Count', fontsize = 20)

ax.set_xlabel('Price', fontsize = 20)

plt.show()
data['Size2'] = round(data['Size']/1000000,1)
bold('**MOST OF THE APPS HAVE SIZE BETWEEN 0 TO 500 MEGABYTES**')

plt.rcParams['figure.figsize'] = (18, 10)

ax = sns.kdeplot(data['Size2'], shade = True, linewidth = 5, color = 'teal')

ax.set_ylabel('Count', fontsize = 20)

ax.set_xlabel('Size', fontsize = 20)

plt.show()
Size_Less_250MB = data[data.Size2 <250]

Size_More_250MB = data[(data.Size2 >=250) & (data.Size2 <1000)]

Size_More_1GB = data[data.Size2 >=1000]
f, axes = plt.subplots (1,3, figsize=(18,10))

ax1 = sns.distplot(Size_Less_250MB.Size2, bins= 20, kde=False,ax=axes[0], color = 'teal')

ax2 = sns.distplot(Size_More_250MB.Size2, bins= 20, kde=False,ax=axes[1], color = 'teal')

ax3 = sns.distplot(Size_More_1GB.Size2, bins= 20, kde=False,ax=axes[2], color = 'teal')

ax1.set(xlabel='Game Size in MB',ylabel='Number of Games')

ax2.set(xlabel='Game Size in MB',ylabel='Number of Games')

ax3.set(xlabel='Game Size in MB',ylabel='Number of Games')

axes[0].set_title('No. of Games Below 250MB')

axes[1].set_title('No. of Games B/W 250MB and 1GB')

axes[2].set_title('No. of Games Above 1GB')

plt.subplots_adjust(wspace=0.2)

plt.show()
plt.rcParams['figure.figsize'] = (18, 10)

data.Developer.value_counts()[:20].plot(kind='bar',color = 'gray', alpha = 0.7, linewidth=4, edgecolor= 'black')

plt.xlabel("Developers", fontsize=20)

plt.ylabel("Count", fontsize=20)

plt.title("TOP 20 Most Commmon Developers ", fontsize=22)

plt.xticks(rotation=90, fontsize = 13) 

plt.show()
bold('**MOST COMMON NAME USED BY GAME DEVELOPERS ARE BATTLE, WAR, PUZZLE**')

fig, ax = plt.subplots(1, 2, figsize=(16,32))

wordcloud = WordCloud(background_color='white',width=800, height=800).generate(' '.join(data['Name']))

wordcloud_sub = WordCloud(background_color='white',width=800, height=800).generate(' '.join(data['Subtitle'].dropna().astype(str)) )

ax[0].imshow(wordcloud)

ax[0].axis('off')

ax[0].set_title('Wordcloud(Name)')

ax[1].imshow(wordcloud_sub)

ax[1].axis('off')

ax[1].set_title('Wordcloud(Subtitle)')

plt.show()
'''A Function To Plot Pie Plot using Plotly'''



def pie_plot(cnt_srs, colors, title):

    labels=cnt_srs.index

    values=cnt_srs.values

    trace = go.Pie(labels=labels, 

                   values=values, 

                   title=title, 

                   hoverinfo='percent+value', 

                   textinfo='percent',

                   textposition='inside',

                   hole=0.7,

                   showlegend=True,

                   marker=dict(colors=colors,

                               line=dict(color='#000000',

                                         width=2),

                              )

                  )

    return trace



bold('**MOST OF THE APPS HAVE 4+ AGE RATING**')

py.iplot([pie_plot(data['Age Rating'].value_counts(), ['cyan', 'gold', 'red'], 'Age Rating')])
bold('**ENTERTAINMENT AND PUZZULE ARE THE MOST POPULAR GAME TPYE IN STRATEGY GAME**')

import squarify



data['Genreslist'] = data['Genres'].str.extract('([A-Z]\w{5,})', expand=True)

temp_df = data['Genreslist'].value_counts().reset_index()



sizes=np.array(temp_df['Genreslist'])

labels=temp_df['index']

colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

plt.figure(figsize=(12,8), dpi= 100)

squarify.plot(sizes=sizes, label=labels, color = colors, alpha=.5, edgecolor="black", linewidth=3, text_kwargs={'fontsize':15})

plt.title('Treemap of Strategy Game Genres on App Store', fontsize = 12)

plt.axis('off')

plt.show()
bold('**THERE IS NO RELATIONSHIP BETWEEN PRICE AND AVERGE USER RATING**')

plt.figure(figsize=(18,10), dpi= 100)

ax = sns.regplot(data=data, x='Price', y='Average User Rating', color = 'teal')

ax.set_ylabel('Average User Rating', fontsize = 20)

ax.set_xlabel('Price', fontsize = 20)

plt.show()
bold('**THERE IS LITTLE POSITIVE RELATIONSHIP BETWEEN SIZE AND AVERGE USER RATING**')

plt.figure(figsize=(18,10), dpi= 100)

ax = sns.regplot(data=data, x='Size', y='Average User Rating', color = 'darkred')

ax.set_ylabel('Average User Rating', fontsize = 20)

ax.set_xlabel('Size', fontsize = 20)

plt.show()
bold('**AVERAGE RATING BASED BY AGE GROUP**')

ax = sns.FacetGrid(data, col="Age Rating", col_wrap=2, height=6, aspect=2,  sharey=False)

ax.map(sns.countplot, 'Average User Rating', color="teal", alpha = 0.7, linewidth=4, edgecolor= 'black')

plt.subplots_adjust(hspace=0.45)

plt.show()
bold('**AVERAGE RATING DISTRIBUTION BY APP SIZE GROUP**')

f, axes = plt.subplots (1,3, figsize=(18,10))

ax1 = sns.boxplot( data = Size_Less_250MB, x = 'Average User Rating', y = 'Size2',  ax=axes[0],color = 'teal')

ax2 = sns.boxplot( data = Size_More_250MB, x = 'Average User Rating', y = 'Size2', ax=axes[1], color = 'teal')

ax3 = sns.boxplot( data = Size_More_1GB, x = 'Average User Rating', y = 'Size2', ax=axes[2], color = 'teal')

ax1.set(xlabel='Average User Rating',ylabel='Game Size in MB')

ax2.set(xlabel='Average User Rating',ylabel='Game Size in MB')

ax3.set(xlabel='Average User Rating',ylabel='Game Size in MB')

axes[0].set_title('No. of Games Below 250MB')

axes[1].set_title('No. of Games Between 250MB and 1GB')

axes[2].set_title('No. of Games Above 1GB')

plt.subplots_adjust(wspace=0.2)

plt.show()
paid = data[data['Price']>0]

free = data[data['Price']==0]

fig, ax = plt.subplots(1, 2, figsize=(15,8))

sns.countplot(data=paid, y='Average User Rating', ax=ax[0], palette='plasma')

ax[0].set_title('Paid Games')

ax[0].set_xlim([0, 1000])



sns.countplot(data=free, y='Average User Rating', ax=ax[1], palette='viridis')

ax[1].set_title('Free Games')

ax[1].set_xlim([0,2000])

plt.tight_layout();

plt.show()
data["Original Release Date"] = pd.to_datetime(data["Original Release Date"])

data["year"] = data["Original Release Date"].dt.year
bold('**SIZE OF THE APPS IS INCREASING OVER TIME**')

plt.rcParams['figure.figsize'] = (18,10)

temp_df = data.groupby(['Original Release Date']).Size.sum().reset_index()

ax = sns.lineplot(data = temp_df, x = 'Original Release Date', y = 'Size', color = 'cornflowerblue')

plt.xlabel('Original Release year', fontsize = 15)

plt.ylabel('Size')

plt.show()
data["Current Version Release Date"] = pd.to_datetime(data["Current Version Release Date"])

data["month"] = data["Current Version Release Date"].dt.month_name() 
bold('**MOST GAMES HAVE BEEN UPDATED IN JULY**')

plt.rcParams['figure.figsize'] = (18, 15)

ax = sns.boxplot(data = data, x ='month', y = np.log1p(data['User Rating Count']), color = 'skyblue')

ax.set_xlabel('Month', fontsize = 20)

ax.set_ylabel('User Rating Count', fontsize = 20)

plt.show()
bold('**4+ AGE RATING IS HAVING THE HIGEST PROPORTION OF APPS**')

top_genres = list(data["Primary Genre"].value_counts().head(10).index)

ct_genre_agerating = pd.crosstab(data[data["Primary Genre"].isin(top_genres)]["Primary Genre"], data["Age Rating"], normalize=0)

ct_genre_agerating.plot.bar(stacked=True, figsize=(18,10))

plt.title("Primary Genre repartition by Age Rating", fontsize = 15)

plt.show()
bold('**NO SINGNIFICANT CHANGE IN THE PROPORTION OF PRICE RANGE IN ALL AGE RATING**')

data["Price Range"] = data["Price"].dropna().map(lambda x: "Free" if x == 0.00 else("Low Price" if 0.99 <= x <= 4.99 else("Medium Price" if 5.99 <= x <= 19.99 else "High Price")))



ct_agerating_pricerange = pd.crosstab(data["Age Rating"], data["Price Range"], normalize=0)

ct_agerating_pricerange.plot.bar(stacked=True, figsize=(18,10))

plt.xticks(rotation=0)

plt.title("Age Rating repartioned by Price Range", fontsize = 15)

plt.show()
plt.rcParams['figure.figsize'] = (18,10)

sns.heatmap(data.corr(), vmin=-1, vmax=1, center=0,

            square=True, annot = True, cmap = 'RdYlGn')

plt.show()
data['GenreList'] = data['Genres'].apply(lambda s : s.replace('Games','').replace('&',' ').replace(',', ' ').split()) 

data['GenreList'].head()
from sklearn.preprocessing import MultiLabelBinarizer # Similar to One-Hot Encoding



test = data['GenreList']

mlb = MultiLabelBinarizer()

res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_, index=test.index)
corr = res.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 14))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
import networkx as nx



stocks = corr.index.values

cor_matrix = np.asmatrix(corr)

G = nx.from_numpy_matrix(cor_matrix)

G = nx.relabel_nodes(G,lambda x: stocks[x])

G.edges(data=True)



def create_corr_network(G, corr_direction, min_correlation):

    H = G.copy()

    for stock1, stock2, weight in G.edges(data=True):

        if corr_direction == "positive":

            if weight["weight"] <0 or weight["weight"] < min_correlation:

                H.remove_edge(stock1, stock2)

        else:

            if weight["weight"] >=0 or weight["weight"] > min_correlation:

                H.remove_edge(stock1, stock2)

                

    edges,weights = zip(*nx.get_edge_attributes(H,'weight').items())

    weights = tuple([(1+abs(x))**2 for x in weights])

    d = nx.degree(H)

    nodelist, node_sizes = zip(*d)

    positions=nx.circular_layout(H)

    

    plt.figure(figsize=(10,10), dpi=72)



    nx.draw_networkx_nodes(H,positions,node_color='#DA70D6',nodelist=nodelist,

                           node_size=tuple([x**2 for x in node_sizes]),alpha=0.8)

    

    nx.draw_networkx_labels(H, positions, font_size=8, 

                            font_family='sans-serif')

    

    if corr_direction == "positive": edge_colour = plt.cm.GnBu 

    else: edge_colour = plt.cm.PuRd

        

    nx.draw_networkx_edges(H, positions, edge_list=edges,style='solid',

                          width=weights, edge_color = weights, edge_cmap = edge_colour,

                          edge_vmin = min(weights), edge_vmax=max(weights))

    plt.axis('off')

    plt.show() 

    

create_corr_network(G, 'positive', 0.3)

create_corr_network(G, 'positive', -0.3)
review = data.sort_values(by='User Rating Count', ascending=False)[['Name', 'Price', 'Average User Rating', 'Size', 'User Rating Count', 'Icon URL']].head(10)

review.iloc[:, 0:-1]
import urllib.request

from PIL import Image



plt.figure(figsize=(6,3))

plt.subplot(131)

image = Image.open(urllib.request.urlopen(review.iloc[0,-1]))

plt.title('1. Clash Of Clans')

plt.imshow(image)

plt.axis('off')



plt.subplot(132)

image = Image.open(urllib.request.urlopen(review.iloc[1,-1]))

plt.title('2.Clash Royale')

plt.imshow(image)

plt.axis('off')



plt.subplot(133)

image = Image.open(urllib.request.urlopen(review.iloc[2,-1]))

plt.title('3. PUBG Mobile')

plt.imshow(image)

plt.axis('off')



plt.show()
data.dropna(inplace = True)

price = data.sort_values(by='Price', ascending=False)[['Name', 'Price', 'Average User Rating', 'Size', 'Icon URL']].head(10)

price.iloc[:, 0:-1]
import urllib.request

from PIL import Image



plt.figure(figsize=(6,3))

plt.subplot(121)

image = Image.open(urllib.request.urlopen(price.iloc[0,-1]))

plt.title('1. Finabase: realtime stocks')

plt.imshow(image)

plt.axis('off')



plt.subplot(122)

image = Image.open(urllib.request.urlopen(price.iloc[1,-1]))

plt.title('2. Tarot - Single and Multiplayer ')

plt.imshow(image)

plt.axis('off')





plt.show()
best = data.sort_values(by=['Average User Rating', 'User Rating Count'], ascending=False)[['Name', 'Average User Rating', 'User Rating Count', 'Size', 

                                                                                         'Price', 'Developer',  'Icon URL',]].head(10)

best.iloc[:, 0:-1]
bold('**Cash, Inc. Fame & Fortune Game Develop by Lion Studios**')

plt.figure(figsize=(5,5))

image = Image.open(urllib.request.urlopen(best.iloc[0, -1]))

plt.axis('off')

plt.title('Cash, Inc. Fame & Fortune Game')

plt.imshow(image)

plt.show()

bold('**--Cash, Inc. Fame & Fortune Game turns out to be best overall game with 5.0 rating and 374772 reviews -- There are also a lot of other Games with 5.0 rating and healthy number of reviews**')