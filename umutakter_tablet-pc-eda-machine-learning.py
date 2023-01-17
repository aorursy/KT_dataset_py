import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import missingno as mno



from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score



from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px



from sklearn import preprocessing



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dfMain=pd.read_csv("../input/tabletpc-priceclassification/tablet.csv")

df=dfMain.copy()
df.head()
df.tail()
df.sample(5)
df.shape
df.info()
df.isna().sum()
df.describe().T
mno.matrix(df, figsize = (20, 6))

plt.show()
BosDegerler = df[df.isna().any(axis=1)]

BosDegerler
corr = df.corr()

f,ax=plt.subplots(figsize=(25,15))

sns.heatmap(corr, annot=True, ax=ax)

plt.show()
plt.figure(figsize=(18,5))

ax = sns.swarmplot(x="ArkaKameraMP", y="OnKameraMP", data=df)

data1=go.Violin(y=df.OnKameraMP,x=df.FiyatAraligi, box_visible=True, line_color='black',

                               meanline_visible=True, fillcolor='lightseagreen', opacity=0.6,

                               x0='Ön Kamera')

layout=dict(title='Ön Kamera İstatistikleri',

           xaxis=dict(title='Fiyat Aralığı',ticklen=5,zeroline=False)

           )

fig = go.Figure(data=data1,layout=layout)

iplot(fig)
df[df.OnKameraMP.isna()==True]
ArkaKameraDegerleri=list(df[df.OnKameraMP.isna()==True].ArkaKameraMP)

ArkaKameraDegerleri
for each in ArkaKameraDegerleri:

    

    OnKameraDegeri = (df[df.ArkaKameraMP==each].OnKameraMP).median()

    

    df.loc[(df['ArkaKameraMP'] == each)&(df.OnKameraMP.isna()), 'OnKameraMP'] = OnKameraDegeri
df.isna().sum()
data2=go.Violin(y=df.RAM,x=df.FiyatAraligi, box_visible=True, line_color='black',

                               meanline_visible=True, fillcolor='lightseagreen', opacity=0.6,

                               x0='Total Bill')

layout=dict(title='RAM İstatistikleri',

           xaxis=dict(title='Fiyat Aralığı',ticklen=5,zeroline=False)

           )

fig = go.Figure(data=data2,layout=layout)

iplot(fig)
df[df.RAM.isna()==True].FiyatAraligi.unique()
df[df.RAM.isna()==True]
df.RAM.fillna(round(df[df.FiyatAraligi=="Pahalı"].RAM.median()),inplace=True)
df.isna().sum()
df.replace("Yok",0,inplace=True)

df.replace("Var",1,inplace=True)
df.head()
trace1=go.Bar(

    x=df[df.FiyatAraligi=="Çok Ucuz"].Renk.value_counts().index,

    y=df[df.FiyatAraligi=="Çok Ucuz"].Renk.value_counts().values,

    name="Çok Ucuz",

    marker=dict(color="rgba(51, 102, 255,0.5)",

               line=dict(color="rgba(0,0,0)",width=1.5)

               ),

    )



trace2=go.Bar(

    x=df[df.FiyatAraligi=="Ucuz"].Renk.value_counts().index,

    y=df[df.FiyatAraligi=="Ucuz"].Renk.value_counts().values,

    name="Ucuz",

    marker=dict(color="rgba(0, 204, 102,0.5)",

               line=dict(color="rgba(0,0,0)",width=1.5)

               ),

    )



trace3=go.Bar(

    x=df[df.FiyatAraligi=="Normal"].Renk.value_counts().index,

    y=df[df.FiyatAraligi=="Normal"].Renk.value_counts().values,

    name="Normal",

    marker=dict(color="rgba(255, 255, 77,0.5)",

               line=dict(color="rgba(0,0,0)",width=1.5)

               ),

    )



trace4=go.Bar(

    x=df[df.FiyatAraligi=="Pahalı"].Renk.value_counts().index,

    y=df[df.FiyatAraligi=="Pahalı"].Renk.value_counts().values,

    name="Pahalı",

    marker=dict(color="rgba(255, 102, 153,0.5)",

               line=dict(color="rgba(0,0,0)",width=1.5)

               ),

    )





data=[trace1,trace2,trace3,trace4]

layout= go.Layout(barmode="group")



fig=go.Figure(data=data,layout=layout)

iplot(fig)
df.drop(['Renk'], axis=1,inplace=True)

df.head()
dfMain=df.copy()

dfMain.replace("Çok Ucuz",0,inplace=True)

dfMain.replace("Ucuz",1,inplace=True)

dfMain.replace("Normal",2,inplace=True)

dfMain.replace("Pahalı",3,inplace=True)
corr = dfMain.corr()

f,ax=plt.subplots(figsize=(25,15))

sns.heatmap(corr, annot=True, ax=ax)

plt.show()
sns.jointplot("FiyatAraligi", "RAM", data=dfMain,

                  kind="kde", space=0, color="g")

plt.show()
fiyatAraliklari=[0,1,2,3]

ramOrtalamasi=[dfMain[dfMain.FiyatAraligi==0].RAM.mean(),dfMain[dfMain.FiyatAraligi==1].RAM.mean(),dfMain[dfMain.FiyatAraligi==2].RAM.mean(),dfMain[dfMain.FiyatAraligi==3].RAM.mean()]

fig = px.bar( x=fiyatAraliklari, y=ramOrtalamasi, height=400, color=ramOrtalamasi)

fig.show()
liste=[]

for i in range(2000):

    liste.append(i)

data=[

    {

        "y":dfMain.RAM,

        "x":liste,

        "mode":"markers",

        "marker":{

            "color":dfMain.FiyatAraligi,

            "showscale":True

        },

        "text":dfMain.FiyatAraligi

    }

]

iplot(data)
dfMain.head()
batayaGucuOrtalamasi=[dfMain[dfMain.FiyatAraligi==0].BataryaGucu.mean(),dfMain[dfMain.FiyatAraligi==1].BataryaGucu.mean(),dfMain[dfMain.FiyatAraligi==2].BataryaGucu.mean(),dfMain[dfMain.FiyatAraligi==3].BataryaGucu.mean()]

fig = px.bar( x=fiyatAraliklari, y=batayaGucuOrtalamasi, height=400, color=batayaGucuOrtalamasi)

fig.show()
cozunurlukYukseklikOrtalamasi=[dfMain[dfMain.FiyatAraligi==0].CozunurlukYükseklik.mean(),dfMain[dfMain.FiyatAraligi==1].CozunurlukYükseklik.mean(),dfMain[dfMain.FiyatAraligi==2].CozunurlukYükseklik.mean(),dfMain[dfMain.FiyatAraligi==3].CozunurlukYükseklik.mean()]

fig = px.bar( x=fiyatAraliklari, y=cozunurlukYukseklikOrtalamasi, height=400, color=cozunurlukYukseklikOrtalamasi)

fig.show()
cozunurlukGenislikOrtalamasi=[dfMain[dfMain.FiyatAraligi==0].CozunurlukYükseklik.mean(),dfMain[dfMain.FiyatAraligi==1].CozunurlukYükseklik.mean(),dfMain[dfMain.FiyatAraligi==2].CozunurlukYükseklik.mean(),dfMain[dfMain.FiyatAraligi==3].CozunurlukYükseklik.mean()]

fig = px.bar( x=fiyatAraliklari, y=cozunurlukGenislikOrtalamasi, height=400, color=cozunurlukGenislikOrtalamasi)

fig.show()
plt.figure(figsize=(15,5))

sns.regplot(x=dfMain.OnKameraMP, y=dfMain.ArkaKameraMP, color="g")

plt.title("ArkaKameraMP-OnKameraMP Regression Plot")

plt.show()
ikisideYok=dfMain[(dfMain['3G']==0)&(dfMain['4G']==0)].shape[0]

BiriVar=dfMain[(dfMain['3G']==0)&(dfMain['4G']==1)].shape[0]+dfMain[(dfMain['3G']==1)&(dfMain['4G']==0)].shape[0]

ikisideVar=dfMain[(dfMain['3G']==1)&(dfMain['4G']==1)].shape[0]

listeVarYok=[ikisideYok,BiriVar,ikisideVar]

labels=['3G ve 4G yok','3G ve 4G den biri var','3G ve 4G de bulunanlar']

trace1={

    "values":listeVarYok,

    "labels":labels,

    "name":"3G ve 4G Bulundurma Durumu",

    "hoverinfo":"label+percent+name",

    "hole":.3,#ortada boşluk bırakıt

    "type":"pie"

}

data=[trace1];

layout={

    "title":"3G ve 4G Bulundurma Durumu",

    "annotations":[

        {"font":{"size":20},

        "showarrow":False,

        "text":"",

        "x":0.20,

        "y":1

        },

    ]

};

fig=go.Figure(data=data,layout=layout)

iplot(fig)
y = df['FiyatAraligi']

x = df.drop(['FiyatAraligi'], axis=1)
y
sns.countplot(x="FiyatAraligi", data=df)

plt.show()
x
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)
x_train
x_test
y_train
y_test
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train,y_train)
y_pred = nb.predict(x_test)
accuracy_score(y_test, y_pred)
cross_val_score(nb, x_test, y_test, cv = 10).mean()
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
report=classification_report(y_test, y_pred)

print(report)
from numpy import set_printoptions

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif
test = SelectKBest(k = 11)

fit = test.fit(x, y)

for indis, skor in enumerate(fit.scores_):

    print(skor, " -> ", x.columns[indis])
y1 = df['FiyatAraligi']

x1 = df[["BataryaGucu", "4G", "DahiliBellek", "Kalinlik", "Agirlik", "CekirdekSayisi", "CozunurlukYükseklik","CozunurlukGenislik","RAM","Dokunmatik"]]
x1_train, x1_test, y1_train, y1_test = train_test_split(x1,

                                                    y1, 

                                                    test_size = 0.25, 

                                                    random_state = 42)
nb = GaussianNB()

nb.fit(x1_train, y1_train)

y1_pred = nb.predict(x1_test)

accuracy_score(y1_test, y1_pred)
reportNB=classification_report(y1_test, y1_pred)

print(reportNB)
cross_val_score(nb, x1_test, y1_test, cv = 10).mean()
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state = 42, criterion='gini')

dtEntropy = DecisionTreeClassifier(random_state = 42, criterion='entropy')

dt_model = dt.fit(x_train, y_train)
from sklearn.tree.export import export_text

r = export_text(dt, feature_names = ['BataryaGucu', 'Bluetooth', 'MikroislemciHizi', 'CiftHat', 'OnKameraMP',

       '4G', 'DahiliBellek', 'Kalinlik', 'Agirlik', 'CekirdekSayisi',

       'ArkaKameraMP', 'CozunurlukYükseklik', 'CozunurlukGenislik', 'RAM',

       'BataryaOmru', '3G', 'Dokunmatik', 'WiFi'])



print(r)
from sklearn.tree import export_graphviz

from sklearn import tree

from IPython.display import SVG

from graphviz import Source

from IPython.display import display



graph = Source(tree.export_graphviz(dt, out_file = None, feature_names = x.columns, filled = True))

display(SVG(graph.pipe(format = 'svg')))
y_pred = dt.predict(x_test)

accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
cross_val_score(dt_model, x, y, cv = 10).mean()
print(classification_report(y_test, y_pred))
cross_val_score(dt_model, x1, y1, cv = 10).mean()
clf_entropy = DecisionTreeClassifier( 

            criterion = "entropy", random_state = 100, 

            max_depth = 3, min_samples_leaf = 5) 

clf_entropy.fit(x1_train, y1_train) 



cross_val_score(dtEntropy, x1, y1, cv = 10).mean()



reportDT=classification_report(y1_test, y1_pred)

print(reportDT)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=400,random_state=1,criterion = "entropy")

rf.fit(x1_train,y1_train)

rf.score(x1_test,y1_test)

y1_pred = rf.predict(x1_test)

cross_val_score(rf, x1, y1, cv = 10).mean()
reportRF=classification_report(y1_test, y1_pred)

print(reportRF)
from numpy import set_printoptions

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)

knn.score(x_test,y_test)
cross_val_score(knn, x, y, cv = 10).mean()
knn_params = {"n_neighbors": np.arange(1,15)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, knn_params, cv = 3)

knn_cv.fit(x_train, y_train)
print("En iyi skor: " + str(knn_cv.best_score_))

print("En iyi parametreler: " + str(knn_cv.best_params_))
score_list=[]

for each in range(1,15):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))



plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuaracy")

plt.show()
knn= KNeighborsClassifier(n_neighbors = 11) # n_neighbors = K

knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)

knn.score(x_test,y_test)
cross_val_score(knn, x, y, cv = 10).mean()
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
reportKNN=classification_report(y_test, y_pred)

print(reportKNN)
print(reportNB)
print(reportDT)
print(reportRF)
print(reportKNN)