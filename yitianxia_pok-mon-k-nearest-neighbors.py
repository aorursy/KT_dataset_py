import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from IPython.display import HTML, Image
pkmn = pd.read_csv("../input/Pokemon.csv", index_col=0)
pkmn.head()
pkmn.isnull().sum()
# Almost half of the Type 2 attribute is empty but it's because many pokemon have only one type. So fill NaN with 'Blank'
pkmn = pkmn.fillna(value={'Type 2':'Blank'})
pkmn.head()
#Rename Column "Type 1" and "Type 2" to plot pie charts
pkmn.rename(columns={"Type 1":"Type1", "Type 2": "Type2"}, inplace=True)
pkmn.head()
d_frame = pkmn.iloc[:800,:]
donut= d_frame.Type1.value_counts()
labels = d_frame.Type1.value_counts().index

#Creat figure
fig = {
    "data":
    [
        {
            "values": donut,
            "labels": labels,
            "domain": {"x": [0, 1]},
            "name": "Type 1",
            "hoverinfo": "label+percent+name",
            "hole": .4,
            "type": "pie"
        }, 
    ],
    "layout":
    {
        "title":"Type 1 of Pokemons",
        "annotations":
        [
            { 
                "font":{"size":20},
                "showarrow":False,
                "text": "",
                "x": 0,
                "y": 1
            },
        ]
    }
}
iplot(fig)
d_frame = pkmn.iloc[:800,:]
donut= d_frame.Type2.value_counts()
labels = d_frame.Type2.value_counts().index

#Creat figure
fig = {
    "data":
    [
        {
            "values": donut,
            "labels": labels,
            "domain": {"x": [0, 1]},
            "name": "Type 2",
            "hoverinfo": "label+percent+name",
            "hole": .4,
            "type": "pie"
        }, 
    ],
    "layout":
    {
        "title":"Type 2 of Pokemons",
        "annotations":
        [
            { 
                "font":{"size":20},
                "showarrow":False,
                "text": "",
                "x": 0,
                "y": 1
            },
        ]
    }
}
iplot(fig)
d_frame = pkmn.iloc[:800,:]
donut= d_frame.Generation.value_counts()
labels = d_frame.Generation.value_counts().index

#Creat figure
fig = {
    "data":
    [
        {
            "values": donut,
            "labels": labels,
            "domain": {"x": [0, 1]},
            "name": "Generation",
            "hoverinfo": "label+percent+name",
            "hole": .4,
            "type": "pie"
        }, 
    ],
    "layout":
    {
        "title":"Generation of Pokemons",
        "annotations":
        [
            { 
                "font":{"size":20},
                "showarrow":False,
                "text": "",
                "x": 0,
                "y": 1
            },
        ]
    }
}
iplot(fig)
hist_data = [pkmn['HP'],pkmn['Attack'],pkmn['Defense'],pkmn['Sp. Atk'],pkmn['Sp. Def'],pkmn['Speed']]
group_labels = list(pkmn.iloc[:,4:10].columns)

fig = ff.create_distplot(hist_data, group_labels, bin_size=5)
iplot(fig, filename='Distribution of Pokemons Stats')
data = []
for i in range(4,10):
    trace = {
            "type": 'violin',
            "x": max(pkmn.iloc[:,i]),
            "y": pkmn.iloc[:,i],
            "name": list(pkmn.columns)[i],
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            }
        }
    data.append(trace)
        
fig = {
    "data": data,
    "layout" : {
        "title": "Violin plot of all stats",
        "yaxis": {
            "zeroline": False,
        }
    }
}

iplot(fig, filename='violin', validate = False)
#Select first 400 pokemons, create 3d scatterplot of their abilities and use HP as solorscale
d_frame = pkmn.iloc[:400,:]
c = d_frame.HP
trace = go.Scatter3d(
    x=d_frame.Attack,
    y=d_frame.Defense,
    z=d_frame.Speed,
    text= d_frame.Name,
    mode='markers',
    marker=dict(
        size=5,
        color= c,           
        colorscale='Viridis',  
        opacity=0.7
    )
)
data = [trace]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Pokemon Ability Correlation Heatmap")
corr = pkmn.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           cmap="BuGn",
           annot =True)
One = pkmn.loc[pkmn.Generation ==1,"Total"]
Two = pkmn.loc[pkmn.Generation ==2,"Total"]
Three = pkmn.loc[pkmn.Generation ==3,"Total"]
Four = pkmn.loc[pkmn.Generation ==4,"Total"]
Five = pkmn.loc[pkmn.Generation ==5,"Total"]
Six = pkmn.loc[pkmn.Generation ==6,"Total"]
plt.figure(figsize=(15,8))
sns.kdeplot(One, color="lightcoral", shade=False)
sns.kdeplot(Two, color="steelblue", shade=False)
sns.kdeplot(Three, color="darkturquoise", shade=False)
sns.kdeplot(Four, color="forestgreen", shade=False)
sns.kdeplot(Five, color="dimgray", shade=False)
sns.kdeplot(Six, color="yellow", shade=False)
plt.legend(['One', 'Two','Three','Four','Five','Six'])
plt.title('Density Plot of Generation')
plt.show()
legendary = pkmn.loc[pkmn.Legendary ==True,"Total"]
not_legendary = pkmn.loc[pkmn.Legendary ==False,"Total"]

plt.figure(figsize=(15,8))
sns.kdeplot(legendary, color="lightcoral", shade=False)
sns.kdeplot(not_legendary, color="forestgreen", shade=False)
plt.legend(['Legendary', 'Not Legendary'])
plt.title('Density Plot of Legendary or Not')
plt.show()
g = sns.pairplot(pkmn, hue='Type1', palette='muted')
pkmn_knn= pkmn.copy()
pkmn_knn.drop(['Name','Type1', 'Type2', 'Total'],axis=1, inplace=True)
pkmn_knn['Legendary'] = pkmn_knn['Legendary'].astype(int)
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(pkmn_knn.drop('Legendary', axis=1))
scaled_data= scaler.transform(pkmn_knn.drop('Legendary', axis=1))
scaled= pd.DataFrame(scaled_data, columns=pkmn_knn.columns[:-1])
X= scaled
y= pkmn_knn['Legendary']
scaled.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=88)
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
predictions= knn.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def get_metrics(y_test, predictions):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, predictions, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, predictions, pos_label=None,
                              average='weighted')
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, predictions, pos_label=None, average='weighted')
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = get_metrics(y_test, predictions)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=30)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)

    return plt
cm= confusion_matrix(y_test, predictions)
fig = plt.figure(figsize=(8, 8))
plot = plot_confusion_matrix(cm, classes=['Non-Legendary','Legendary'], normalize=False, title='Confusion Matrix')
plt.show()
error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
