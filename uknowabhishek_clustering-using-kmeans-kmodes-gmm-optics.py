import numpy as np

import pandas as pd

import re

import seaborn as sns

import matplotlib.pyplot as plt



import warnings



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler



from sklearn.mixture import GaussianMixture as GMM

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans



from sklearn.metrics import adjusted_rand_score,calinski_harabasz_score

from sklearn.metrics import davies_bouldin_score,completeness_score,homogeneity_score,v_measure_score

from nltk.stem import WordNetLemmatizer



from IPython.core.interactiveshell import InteractiveShell

from IPython.core.display import display, HTML

pd.options.display.max_rows = 500

InteractiveShell.ast_node_interactivity = "all"

warnings.filterwarnings("ignore")
df= pd.read_csv('/kaggle/input/advertising/advertising.csv',parse_dates = ['Timestamp'])

display(HTML('<h2 id = "inf">Basic Information about data like data-type and count</h2>'))

df.info()

display(HTML('<h2>Statistical summary of data</h2>'))



df.describe()

display(HTML('<h2>Random Sample size of 5 observations</h2>'))



df.sample(5).T
display(HTML('<h2 id="uni">Unique Values in dataset</h2>'))

df.nunique()

display(HTML('<b> Besides these it has <i>'+str(df['Timestamp'].dt.year.nunique())+'</i> year values and <i>'

             +str(df['Timestamp'].dt.month.nunique())+'</i> month values in Timestamp feature</b>'))
f_dct = {n : re.sub('[^A-Za-z0-9]+','_',n) for n in df.columns.values}

df.rename(columns = f_dct,inplace=True)
p=df[['Daily_Time_Spent_on_Site', 'Age', 'Daily_Internet_Usage']].boxplot(figsize = (10,8),grid=True,fontsize=10)

plt.suptitle('Box plot for features',fontsize=15)
pd.crosstab(df.Country,df.Clicked_on_Ad).sort_values(1,ascending=False)
pd.crosstab(df.City,'count').sort_values('count',ascending=False)
plt.figure(figsize=(10, 10))

p = sns.pairplot(df, hue ='Clicked_on_Ad',

    vars=['Daily_Time_Spent_on_Site', 'Age', 'Area_Income', 'Daily_Internet_Usage']

                 ,diag_kind='kde',   palette='bright')

plt.show()
df['hour']=df['Timestamp'].dt.hour

df['day'] = df['Timestamp'].dt.day

df['month'] = df['Timestamp'].dt.month

df['weekday'] = df['Timestamp'].dt.weekday
display(HTML("<h3>Dropping unusable features</h3> We are dropping here ['Timestamp','City','Country']"))

df.drop(columns=['Timestamp','City','Country'],inplace=True)

display(HTML('<b> Now shape of dataset is '+str(df.shape)+'</b>'))

display(HTML('<h3> New sample of data</h3>'))

df.sample(n=5)
X = df[ ['Daily_Time_Spent_on_Site', 'Age', 'Area_Income', 'Daily_Internet_Usage', 'Male','weekday','day','hour','month']]

Y = df[['Clicked_on_Ad']].to_numpy().ravel()

X_scaled = StandardScaler().fit_transform(X.copy())
km = KMeans(n_clusters=2) #K-Means model

cluster_km = km.fit_predict(X_scaled) #fitting means it tries to understand the data and predict will give cluster lables
from kmodes.kmodes import KModes

km1 = KModes(n_clusters=2,init='Cao')

cluster_km1 = km1.fit_predict(X_scaled)
gmm = GMM(n_components=2, covariance_type='full', max_iter=100, n_init=10)

cluster_gmm = gmm.fit_predict(X)
from sklearn.cluster import OPTICS

optics = OPTICS(min_samples=2)

cluster_optics = optics.fit_predict(X_scaled)
models = [km,km1,gmm,optics]

clst = [cluster_km,cluster_km1,cluster_gmm,cluster_optics]

t ='<table border=1 color = "#000000"><tr><th>model</th><th>ARI</th><th>calinski_harabasz_score</th><th>davies_bouldin_score</th>'

t+='<th>completeness_score</th><th> homogeneity_score </th><th>v_measure_score</th>'

for i in range(4):

    t = t+('<tr><td>'+str(models[i])+'</td>'+'<td>'+str(adjusted_rand_score(Y,clst[i]))+'</td>')

    t = t+('<td>'+str(calinski_harabasz_score(X_scaled,clst[i]))+'</td>')

    t = t+('<td>'+str(davies_bouldin_score(X_scaled,clst[i]))+'</td>')

    t = t+('<td>'+str(completeness_score(Y,clst[i]))+'</td>')

    t = t+('<td>'+str(homogeneity_score(Y,clst[i]))+'</td>')

    t = t+('<td>'+str(v_measure_score(Y,clst[i]))+'</td></tr>')

t+='</table>'    

display(HTML(t))
from sklearn.metrics import classification_report

for i in range(4):

    display(HTML('<h4>'+str(models[i])+'</h4>'))

    print(classification_report(Y,clst[i]))
tsne = TSNE(n_components = 2)

tsne_out = tsne.fit_transform(X_scaled)

fig, axs = plt.subplots(2,2, figsize=(15, 15))

plt.suptitle('TSNE Visualisation for different cluster models',fontsize=15)

for i in range(4):

    p = axs[i//2][i%2].scatter(tsne_out[:, 0], tsne_out[:, 1],marker=10,s=10,linewidths=5,c=clst[i])

    axs[i//2][i%2].set_title(models[i])
#from nltk.stem import WordNetLemmatizer

#import nltk

#nltk.download('wordnet')

topics = []

stemmer = WordNetLemmatizer()

for i in range(X.shape[0]):

    topic = re.sub(r'\W',' ',df.Ad_Topic_Line[i])

    topic = re.sub(r'\s+[a-zA-Z]\s+', ' ',topic)

    

    # remove all single characters

    topic = re.sub(r'\s+[a-zA-Z]\s+', ' ', topic)

    

    # Remove single characters from the start

    topic = re.sub(r'\^[a-zA-Z]\s+', ' ', topic) 

    

    # Substituting multiple spaces with single space

    topic = re.sub(r'\s+', ' ', topic, flags=re.I)

    

    # Removing prefixed 'b'

    topic = re.sub(r'^b\s+', '', topic)

    

    # Converting to Lowercase

    topic = topic.lower()

    

    # Lemmatization

    topic = topic.split()



    topic = [stemmer.lemmatize(word) for word in topic]

    topic = ' '.join(topic)

    

    topics.append(topic)



tfidfconverter = TfidfVectorizer( max_features=500,min_df=3 ,max_df=0.8,stop_words='english') #

X = tfidfconverter.fit_transform(topics)
df1 = pd.DataFrame(X.toarray(),columns=tfidfconverter.get_feature_names())

df1 = pd.concat([df,df1],axis='columns')

df1.drop(columns=['Ad_Topic_Line','Clicked_on_Ad'],inplace=True)

X1 = StandardScaler().fit_transform(df1)
clst_txt = [m.fit_predict(X1) for m in models]
t ='<table border=1 color = "#000000"><tr><th>model</th><th>ARI</th><th>calinski_harabasz_score</th><th>davies_bouldin_score</th>'

t+='<th>completeness_score</th><th> homogeneity_score </th><th>v_measure_score</th>'

for i in range(4):

    t = t+('<tr><td>'+str(models[i])+'</td>'+'<td>'+str(adjusted_rand_score(Y,clst[i]))+'</td>')

    t = t+('<td>'+str(calinski_harabasz_score(X_scaled,clst[i]))+'</td>')

    t = t+('<td>'+str(davies_bouldin_score(X_scaled,clst[i]))+'</td>')

    t = t+('<td>'+str(completeness_score(Y,clst[i]))+'</td>')

    t = t+('<td>'+str(homogeneity_score(Y,clst[i]))+'</td>')

    t = t+('<td>'+str(v_measure_score(Y,clst[i]))+'</td></tr>')

t+='</table>'    

display(HTML(t))
for i in range(4):

    print(models[i])

    print(classification_report(Y,clst_txt[i]))
tsne_out = tsne.fit_transform(X1)

fig, axs = plt.subplots(2,2, figsize=(15, 15))

plt.suptitle('TSNE Visualisation for different cluster models',fontsize=15)

for i in range(4):

    p = axs[i//2][i%2].scatter(tsne_out[:, 0], tsne_out[:, 1],marker=10,s=10,linewidths=5,c=clst_txt[i])

    axs[i//2][i%2].set_title(models[i])