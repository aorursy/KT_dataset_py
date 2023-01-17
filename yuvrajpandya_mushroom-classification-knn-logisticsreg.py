import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')
# constants
fontsize = 15
# default plots styles
# plt.rcParams['font.sans-serif'] = 'Arial'
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['text.color'] = '#bdbdbd'
# plt.rcParams['axes.labelcolor']= '#bdbdbd'
# plt.rcParams['xtick.color'] = '#c4c4c4'
# plt.rcParams['ytick.color'] = '#c4c4c4'
# plt.rcParams['font.size']=12
df = pd.read_csv("../input/mushrooms/mushrooms.csv") #https://raw.githubusercontent.com/yuvrajpandya/ScikitLearn/master/Datasets/mushrooms.csv
df.head()
# quick check to see how balanced the dataset is
df['class'].value_counts()
# distribution of the labels
f, ax = plt.subplots(figsize=(6,4))
ax = sns.countplot(x="class", data=df, palette=['#fd4b2a','#65d556'])
ax.set_xticklabels(['poisonous', 'edible']);
# checking for missing data
df.isnull().any()
df.describe().T
# function to prepare bars for any feature to analyze its distribution across edible and poisonous classes
# returns a sorted dataframe with <feature> and count columns
def preparebars(feature, labels):
    # feature = 'cap-shape'
    # labels = {'b':'bell', 'f':'flat', 'k':'knobbed', 's':'sunken', 'x':'convex', 'c':'conical'}

    df_feature = df.groupby(by=['class', feature])['cap-surface'].count().reset_index().rename(columns={'cap-surface':'count'})

    edible_bars = df_feature.loc[df_feature['class']=='e',[feature,'count']].reset_index(drop=True) # dropping index is imp as later we are appending missing cat using .loc method
    poisonous_bars = df_feature.loc[df_feature['class']=='p',[feature,'count']].reset_index(drop=True)
    
    # append missing category's count as 0
    unique_cat = df[feature].unique()
    for cat in unique_cat:
        if cat not in edible_bars[feature].values:
            edible_bars.loc[edible_bars.shape[0]] = [cat, 0]
            
        if cat not in poisonous_bars[feature].values:
            poisonous_bars.loc[poisonous_bars.shape[0]] = [cat,0]
    
    # replace the mnemonics with the full name of the categories of the feature
    for key,val in labels.items():
        edible_bars.replace(to_replace=key, value=val, inplace=True)
        poisonous_bars.replace(to_replace=key, value=val, inplace=True)
    
    # IMPORTANT: the order of the records will not be the same in both dataframes, hence, first sort in desc the edible df and select poisonous df based on the edible df's records ignoring the index
    edible_bars = edible_bars.sort_values(by=['count'], ascending=False, inplace=False).reset_index(drop=True)
    return edible_bars, \
           poisonous_bars.iloc[pd.Index(poisonous_bars[feature]).get_indexer(edible_bars[feature].values)]
def autolabel(bars, axes):
    for bar in bars:
        height = bar.get_height()
        axes.text(bar.get_x() + bar.get_width()/2., 1*height,'%d' % int(height),
                ha='center', va='bottom',fontsize=9)
# shape
edible_shape, poisonous_shape = preparebars('cap-shape', {'b':'bell', 'f':'flat', 'k':'knobbed', 's':'sunken', 'x':'convex', 'c':'conical'})

# plot cap-shape and edible/poisonous mushroom
fig, ax = plt.subplots(figsize=(10,8))
ax.set_title('Edible & Poisonous Mushrooms Based On Cap Shape')
width = 0.3
ed_indexes = [x+1 for x in edible_shape.index]
po_indexes = [x+width for x in ed_indexes]

labels = edible_shape['cap-shape']

# edible mushrooms having different cap-shape
edible_bars = ax.bar(ed_indexes, edible_shape['count'], width, color='#8ff776')

# poisonous mushrooms having different cap-shape
poisonous_bars = ax.bar(po_indexes, poisonous_shape['count'], width, color='#ff5f39')

# set x & y axis ticks & labels
ax.set_xticks([x+width/2 for x in ed_indexes])
ax.set_xticklabels(labels)
ax.set_xlabel('Cap Shape', fontsize = fontsize)
ax.set_ylabel('Mushrooms', fontsize = fontsize)

# set the label for each bar
autolabel(edible_bars, ax)
autolabel(poisonous_bars, ax)

ax.legend(frameon=False, labels=['Edible', 'Poisonous']);
# color
# cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
edible_color, poisonous_color = preparebars('cap-color', {'n':'brown', 'b':'buff', 'c':'cinnamon', 'g':'gray', 'r':'green', 'p':'pink', 'u':'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'})
poisonous_color
# plot cap-color and edible/poisonous mushroom
fig, ax = plt.subplots(figsize=(10,8))
ax.set_title('Edible & Poisonous Mushrooms Based On Cap Color')
width = 0.4
ed_indexes = [x+1 for x in edible_color.index]
po_indexes = [x+width for x in ed_indexes]

labels = edible_color['cap-color']

# edible mushrooms having different cap-color
edible_bars = ax.bar(ed_indexes, edible_color['count'], width, color='#8ff776')

# poisonous mushrooms having different cap-color
poisonous_bars = ax.bar(po_indexes, poisonous_color['count'], width, color='#ff5f39')

# set x & y axis ticks & labels
ax.set_xticks([x+width/2 for x in ed_indexes])
ax.set_xticklabels(labels)
ax.set_xlabel('Cap Color', fontsize = fontsize)
ax.set_ylabel('Mushrooms', fontsize = fontsize)

ax.legend(['Edible', 'Poisonous'])

# label each bar to shpw the count
autolabel(edible_bars, ax)
autolabel(poisonous_bars, ax);
# odor
# odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
edible_odor, poisonous_odor = preparebars('odor', {'a':'almond', 'l':'anise', 'c':'creosote', 'y': 'fishy', 'f':'foul','m':'musty', 'n':'none', 'p':'pungent', 's':'spicy'})

# plot odor and edible/poisonous mushroom
fig, ax = plt.subplots(figsize=(10,8))
ax.set_title('Edible & Poisonous Mushrooms Based On Odor')
width = 0.4
ed_indexes = [x+1 for x in edible_odor.index]
po_indexes = [x+width for x in ed_indexes]

labels = edible_odor['odor']

# edible mushrooms having different cap-color
edible_bars = ax.bar(ed_indexes, edible_odor['count'], width, color='#8ff776')

# poisonous mushrooms having different cap-color
poisonous_bars = ax.bar(po_indexes, poisonous_odor['count'], width, color='#ff5f39')

# set x & y axis ticks & labels
ax.set_xticks([x+width/2 for x in ed_indexes])
ax.set_xticklabels(labels)
ax.set_xlabel('Odor', fontsize = fontsize)
ax.set_ylabel('Mushrooms', fontsize = fontsize)

ax.legend(['Edible', 'Poisonous'])

# label each bar to show the count
autolabel(edible_bars, ax)
autolabel(poisonous_bars, ax);
edible_habitat, poisonous_habitat = preparebars('habitat', {'g':'grasses','l':'leaves','m':'meadows','p':'paths', 'u':'urban','w':'waste','d':'woods'})
edible_habitat['class'] = 'edible'
poisonous_habitat['class'] = 'poisonous'

habitat_merged = edible_habitat.append(poisonous_habitat)
plt.figure(figsize=(10,8))
habitat = habitat_merged.groupby(by=['habitat'])['count'].sum()
plt.pie(habitat, autopct='%.2f%%', 
        labels=['grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods'],
        colors=['#fef1c3', '#73ab7b', '#de6253', '#ffe2ff', '#665ea0', '#ca81ae', '#ec9f6e'],
       center=[0,0]);
plt.title('Mushroom habitats', fontsize=fontsize);
# using plotly express' interactive sunburst chart to show the hierarchical data in donuts charts

fig = px.sunburst(habitat_merged, path=['habitat', 'class'], values='count', color='class',
                 color_discrete_map={'edible':'lightgreen', 'poisonous':'red'})
fig.show()
# split the dataset into features and label
X = df.loc[:, df.columns!='class']
y = df['class']
# one hot encoding using get_dummies
X = pd.get_dummies(X, drop_first=True)

#label encode the labels(response variable)
encode_y = LabelEncoder()
y = encode_y.fit_transform(y)
# 22 features have been split into 95 features
X.head()
# check for poisonous mushrooms labels
y==1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# check the size of the train and test sets
X_train.shape[0]/(X_train.shape[0]+X_test.shape[0])*100
X_test.shape[0]/(X_train.shape[0]+X_test.shape[0])*100
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
# plotting the PCA transformed train set
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(10,6))
    for edibility, col, label in zip((0, 1), ('lightgreen', 'red'), ('edible', 'poisonous')):
        plt.scatter(X_train[y_train==edibility, 0],
                    X_train[y_train==edibility, 1],
                    label=label,
                    c=col)
    plt.title('PCA transformation (n=2) to classify the mushrooms based on their edibility')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
# list of models to train
logisticsRegressionModel = LR()
KNNModel = KNN(n_neighbors=3, n_jobs=-1)
models = [logisticsRegressionModel, KNNModel]
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

cv = KFold(n_splits=10, random_state=42, shuffle=True)

print('Cross validation using KFolds (K=10) | Used for generalization of the model')

for model in models:
    accuracy=[]
    print('\n')
    print(type(model).__name__)
    print('---------------------------------')
    accuracy.append(cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro', verbose=2))
    print('F1 scores are: ', accuracy)
    print('Mean F1 score of the model: ', np.mean(accuracy))
print ("This is base model experiementation to decide benchmark models")
for model in models:
  print('\n')
  print(type(model).__name__)
  print('---------------------------------')
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test,y_pred)
  f1 = f1_score(y_test, y_pred)
  print('Accuracy: ',accuracy)
  print('F1 score: ',f1)  
from mlxtend.plotting import plot_decision_regions
def knn_comparison(X, y, k, axes):
    clf = KNN(n_neighbors=k)
    clf.fit(X, y)
    # Plotting decision region
    plot_decision_regions(X, y, clf=clf, legend=1, ax=axes)
    axes.set_title('KNN with k='+str(k))
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,6))

knn_comparison(X_train, y_train, 3, ax1)
knn_comparison(X_train, y_train, 15, ax2)
# knn_comparison(X_train, y_train, 25, ax3)
# knn_comparison(X_train, y_train, 50, ax4)

for ax in (ax1,ax2):
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in (ax1,ax2):
    ax.label_outer()