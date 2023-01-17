#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from scipy.cluster.hierarchy import dendrogram, linkage
#import Dataset
data = pd.read_csv('../input/dataset.csv')
data3 = data.loc[:,['kume']].values
#rename columns and drop not importance axis
names = ["araba-tur","airbag","araba-yas","araba-performans","araba-kullanım-tur","araba-km",
         "araba-yakıt","araba-segment","araba-parca","arac-hitap",
         "butce","konfor-skorlama","araba-model","marka","sahibinden-link","arabam-link","kume"]

names2 = ["araba-tur","airbag","araba-yas","araba-performans","araba-kullanım-tur","araba-km",
         "araba-yakıt","araba-segment","araba-parca","arac-hitap",
         "butce","konfor-skorlama","marka","kume"]

data = data.rename(columns=dict(zip(data.columns, names)))

data = data.drop(['sahibinden-link'], axis=1)
data = data.drop(['arabam-link'], axis=1)
data = data.drop(['araba-model'], axis=1)
data.head(5)
#Creating Mapping Dictionaries
mapping_tur = {"2. El":0, "Sıfır":1}
mapping_yas = {"0-1":1, "1-3":2, "3-5":3, "5-8":4, "8-10":5, "10-12":6, "12+":7}
mapping_performans = {"Vasat Performans, Az Yakması":0, "Yüksek Performans, Çok Yakması":1, "Standart Performans, Az Yakması":0}
mapping_kullanım = {"Uzun Süre Binme Odaklı":0, "Satıp Para Kazanma Odaklı":1}
mapping_km = {"0-25.000":1, "0-25.002":1, "25.000-50.000":2, "50.000-100.000":3, "100.000-200.000":4, "200.000+":5}
mapping_yakıt = {"LPG":0, "Dizel":1, "Benzinli":2, "Farketmez":3}
mapping_segment = {"A Segmenti (Ekonomik Az Yakanlar, i10)":0, "B Segmenti (Hyundai Getz, Polo)":1, "C Segmenti (Honda Civic, Renault Fluence)":2,
                  "D Segmenti (Mercedes C Serisi, VW Passat, Ford Mondeo)":3, "E Segmenti (BMW 5 serisi, Volvo s80)":4, "F Segmenti (Audi A8, BMW 7 serisi)":5,
                  "G Segmenti (Porshce 911)":6, "J Segmenti (4x4 Jipler vs.)":7, "D Segmenti (Mercedes C Serisi, VW Passat)":3}
mapping_parca = {"Sürekli Sorun Çıkarsın Ucuz Parçaları Olsun":0, "Parçalar Sağlam ve Pahalı Olsun, Az Sorun Çıkarsın.":1, "Arada Bir Sorun Çıkarsın, Ucuz Parçaları Olsun":0}
mapping_hitap = {"Aile Aracı":0, "Ticari":1, "Şahıs Aracı":2, "Off Road":3}
mapping_butce = {"0-15.000":0, "15.000-25.000":1, "25.000-35.000":2, "35.000-45.000":3, "45.000-55.000":4, "55.000-65.000":5,
                "65.000-75.000":6, "75.000-85.000":7, "85.000-100.000":8, "100.000-200.000":9, "200.000+":10}
mapping_marka = {"Alfa Romeo":0, "Audi":1, "Bmw":2, "Chevrolet":3, "Citroen":4, "Dacia":5, "Fiat":6, "Ford":7, "Honda":8,
                "Hyundai":9, "Kia":10, "Mercedes":11, "Mitsubishi":12, "Nissan":13, "Opel":14, "Peugeot":15, "Porsche":16,
                "Renault":17, "Toyota":18, "Volkswagen":19, "Volvo":20, "Skoda":21, "Mazda":22, "Mini":23, "Land Rover":24,
                "Seat":25}

#Mapping on Data
data['araba-tur'] = data['araba-tur'].map(mapping_tur)
data['araba-yas'] = data['araba-yas'].map(mapping_yas)
data['araba-performans'] = data['araba-performans'].map(mapping_performans)
data['araba-kullanım-tur'] = data['araba-kullanım-tur'].map(mapping_kullanım)
data['araba-km'] = data['araba-km'].map(mapping_km)
data['butce'] = data['butce'].map(mapping_butce)
data['araba-yakıt'] = data['araba-yakıt'].map(mapping_yakıt)
data['araba-segment'] = data['araba-segment'].map(mapping_segment)
data['araba-parca'] = data['araba-parca'].map(mapping_parca)
data['arac-hitap'] = data['arac-hitap'].map(mapping_hitap)
data['marka'] = data['marka'].map(mapping_marka)

data8=data
data9=data
data11 = data['marka']
data.head(5)
#I going to define some funcs for Clustering DataSet base on brand

def plotData(data, marka = None):
    if marka != None: 
        data = data[(data.marka == mapping_marka[marka])]
        print("Opinion of ", marka)
    fig = plt.figure(figsize=(25,10))
    
    names = ["araba-tur","airbag","araba-yas","araba-performans","araba-kullanım-tur","araba-km",
         "araba-yakıt","araba-segment","araba-parca","arac-hitap",
         "butce","konfor-skorlama","marka","kume"]
    
    for i in range(0,13):
        p1 = fig.add_subplot(2,5,i-3)
        data[names[i]].value_counts().plot(kind = 'pie', autopct='%.1f%%'); 
        plt.ylabel(" ", fontsize = 15)
        plt.title(Q[i-4])
    plt.grid()
    plt.savefig(marka)
    plt.savefig(marka + ".pdf")
    
def getOpinion(data, marka = None):
    if marka != None: 
        data = data[(data.marka == mapping_marka[marka])]
    return [data[col].mean() for col in names2[0:]]

opinions = dict()
for k in mapping_marka.keys():
    opinions[k] = getOpinion(data, marka = k)

df = pd.DataFrame.from_dict(opinions)
df.rename(index = dict(zip(range(len(names2[0:])),names2[0:])),inplace=True)
df = df.reindex(columns=['Alfa Romeo', 'Audi', 'Bmw', 'Chevrolet', 'Citroen', 'Dacia', 'Fiat', 'Ford', 'Honda', 'Hyundai',
                        'Kia', 'Mercedes', 'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 'Renault', 'Toyota',
                        'Volkswagen', 'Volvo'])
df.T
from sklearn.preprocessing import MinMaxScaler

data9 = data9.drop(['kume'], axis=1)
data9 = data9.drop(['marka'], axis=1)

scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(data9)
data9 = scaler.transform(data9)
data9 = pd.DataFrame(data9)

data9 = pd.concat([data9, data11], axis = 1)

names = ["araba-tur","airbag","araba-yas","araba-performans","araba-kullanım-tur","araba-km",
         "araba-yakıt","araba-segment","araba-parca","arac-hitap",
         "butce","konfor-skorlama","marka"]

data9 = data9.rename(columns=dict(zip(data9.columns, names)))
data9.head(8)
names5 = ["araba-tur","airbag","araba-yas","araba-performans","araba-kullanım-tur","araba-km",
         "araba-yakıt","araba-segment","araba-parca","arac-hitap",
         "butce","konfor-skorlama","marka"]

names4 = ["araba-tur","airbag","araba-yas","araba-performans","araba-kullanım-tur","araba-km",
         "araba-yakıt","araba-segment","araba-parca","arac-hitap",
         "butce","konfor-skorlama","marka"]

def getOpinion2(data9, marka = None):
    if marka != None: 
        data9 = data9[(data9.marka == mapping_marka[marka])]
    return [data9[col].mean() for col in names5[0:]]
opinions = dict()
for k in mapping_marka.keys():
    opinions[k] = getOpinion2(data9, marka = k)

df2 = pd.DataFrame.from_dict(opinions)
df2.rename(index = dict(zip(range(len(names4[0:])),names4[0:])),inplace=True)
df2 = df2.reindex(columns=['Alfa Romeo', 'Audi', 'Bmw', 'Chevrolet', 'Citroen', 'Dacia', 'Fiat', 'Ford', 'Honda', 'Hyundai',
                        'Kia', 'Mercedes', 'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 'Renault', 'Toyota',
                        'Volkswagen', 'Volvo'])
print("")
df2.T
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

names2 = ["araba-tur","airbag","araba-yas","araba-performans","araba-kullanım-tur","araba-km",
         "araba-yakıt","araba-segment","araba-parca","arac-hitap",
         "butce","konfor-skorlama","marka","kume"]

df2 = data8
x = df2.loc[:, names2].values
y = df2.loc[:,['kume']].values
x = StandardScaler().fit_transform(x)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df2[['kume']]], axis = 1)
finalDf.head(8)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier

predictors = finalDf.drop(["kume"], axis=1)
target = finalDf["kume"]
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.25, random_state = 0)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier

models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Decision Tree (CART)',DecisionTreeClassifier())) 
models.append(('K-NN', KNeighborsClassifier()))
models.append(('AdaBoostClassifier', AdaBoostClassifier()))
models.append(('BaggingClassifier', BaggingClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier()))

for name, model in models:
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    from sklearn import metrics
    print("%s -> ACC: %%%.2f" % (name,metrics.accuracy_score(y_test, y_pred)*100))
#Lets Have a Look At Correlations
fig, ax = plt.subplots()
fig.set_size_inches(15,15)
sns.heatmap(data.corr(),cbar=True, annot=True, square=True, annot_kws={'size': 12})
plt.tight_layout()
plt.savefig('2-elaraba-corr.png')
corr = df.corr()
fig, ax = plt.subplots()
fig.set_size_inches(20,20)
mask = np.zeros_like(corr) #eğer corr bozuksa markaları göstermiyorsa bunu 
mask[np.triu_indices_from(mask)] = True #ve bunu silip shift+enter yapın ondan sonra geri yapıştırın ve shift+enter
sns.heatmap(corr, mask=mask, cbar=True, annot=True, square=True, annot_kws={'size': 10})
plt.savefig('car-corr.png')
qs = [q for q in questions.features if q not in ["Sex","Age","Region","Education"]]
qf = df.loc[qs]

fig, ax = plt.subplots(figsize=(20,6))
ax.xaxis.set(ticks=range(0,11), # Manually set x-ticks
ticklabels=qs)
qf[['Alfa Romeo','Audi','Bmw', 'Chevrolet', 'Citroen', 'Dacia', 'Fiat', 'Ford', 'Honda', 'Hyundai',
                        'Kia', 'Mercedes', 'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 'Renault', 'Toyota',
                        'Volkswagen', 'Volvo']].plot(ax=ax,alpha=0.75, rot=80)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid()
plt.savefig('compare.pdf')
df = df.T
df = df.drop(['marka'], axis=1)
df = df.drop(['kume'], axis=1)
df = df.T

#plot data
fig, ax = plt.subplots(figsize=(30,10))
ax.xaxis.set(ticks=range(0,14), # Manually set x-ticks
ticklabels=["araba-tur","airbag","araba-yas","araba-performans","araba-kullanım-tur","araba-km",
         "araba-yakıt","araba-segment","araba-parca","arac-hitap",
         "butce","konfor-skorlama","kume"])
df.plot(ax=ax)
plt.grid()
#Split Data By Train and Test
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier

data = data.drop(['marka'], axis=1)
predictors = data.drop(["kume"], axis=1)
target = data["kume"]
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.25, random_state = 0)
#Le
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier

models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Decision Tree (CART)',DecisionTreeClassifier())) 
models.append(('K-NN', KNeighborsClassifier()))
models.append(('AdaBoostClassifier', AdaBoostClassifier()))
models.append(('BaggingClassifier', BaggingClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier()))

for name, model in models:
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    from sklearn import metrics
    print("%s -> ACC: %%%.2f" % (name,metrics.accuracy_score(y_test, y_pred)*100))
#Lets Look at Feature Importance
rf = RandomForestClassifier(n_estimators=200, random_state=39, max_depth=2)
rf.fit(X_train, y_train)
from sklearn.ensemble import RandomForestClassifier
questions = pd.DataFrame({'features': data.columns[:-1],'importance': rf.feature_importances_})
questions = questions.sort_values(by='importance', ascending=False)
questions
import networkx as nx

#Changes from dataframe to matrix, so it is easier to create a graph with networkx
cor_matrix = np.asmatrix(corr)

#Crates graph using the data of the correlation matrix
G = nx.from_numpy_matrix(cor_matrix)

#relabels the nodes to match the  stocks names
G = nx.relabel_nodes(G,lambda x: df.columns[x])
def drawGraph(G, size = 20):
    fig, ax = plt.subplots()
    fig.set_size_inches(size,size)
    
    pos_fr = nx.fruchterman_reingold_layout(G)
    edges = G.edges()

    weights = [G[u][v]['weight'] for u,v in edges]
    labels = {e: round(G[e[0]][e[1]]['weight'],2) for e in edges}
    weights2 = [w**2 for w in weights]

    nx.draw(G, pos=pos_fr, node_size=1000, node_color='lightblue', with_labels=True)

    # Plot edge labels
    nx.draw_networkx_edge_labels(G, pos=pos_fr, edge_labels=labels)
    plt.savefig('graph.pdf')
    
drawGraph(G)
# remove edges with correlation < 0.5
G.remove_edges_from([(u,v) for u,v,e in G.edges(data = True) if e['weight'] < 0.5])
drawGraph(G, size =30)
# remove edges with correlation < 0.8
G.remove_edges_from([(u,v) for u,v,e in G.edges(data = True) if e['weight'] < 0.8])
drawGraph(G, size=30)
# remove edges with correlation < 0.9
G.remove_edges_from([(u,v) for u,v,e in G.edges(data = True) if e['weight'] < 0.9])
drawGraph(G, size=30)
#Changes from dataframe to matrix, so it is easier to create a graph with networkx
cor_matrix = np.asmatrix(df.T.corr())

#Crates graph using the data of the correlation matrix
G = nx.from_numpy_matrix(cor_matrix)

#relabels the nodes to match the  stocks names
G = nx.relabel_nodes(G,lambda x: df.T.columns[x])
drawGraph(G, size = 25)
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram


#data_array = ((np.float, len(data['marka'].dtype.names)))
data_array = df.transpose()
data_array = np.array(data_array)
data_dist = pdist(data_array) # computing the distance
data_link = linkage(data_dist)
dendrogram(data_link,labels=data_array.dtype.names)
plt.xlabel('Araba Modelleri')
plt.ylabel('Uzaklık')
plt.suptitle('Samples clustering', fontweight='bold', fontsize=14);
# Compute and plot first dendrogram.
fig = plt.figure(figsize=(12,12))
# x ywidth height
ax1 = fig.add_axes([0.05,0.1,0.2,0.6])
Y = linkage(data_dist, method='single')
Z1 = dendrogram(Y, orientation='right',labels=data_array.dtype.names) # adding/removing the axes
ax1.set_xticks([])


# Compute and plot second dendrogram.
ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
Z2 = dendrogram(Y)
ax2.set_xticks([])
ax2.set_yticks([])

#Compute and plot the heatmap
axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
idx1 = Z1['leaves']
idx2 = Z2['leaves']
D = squareform(data_dist)
D = D[idx1,:]
D = D[:,idx2]
im = axmatrix.matshow(D, aspect='auto', origin='lower',cmap=plt.cm.YlGnBu)
axmatrix.set_xticks([])
axmatrix.set_yticks([])

# Plot colorbar.
axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
plt.colorbar(im, cax=axcolor)