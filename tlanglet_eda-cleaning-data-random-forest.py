# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



#sns.set(

#    font_scale=1.5,

#    style="whitegrid",

#    rc={'figure.figsize':(20,7)}

#)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/learn-together/train.csv", index_col='Id')

train_data.sample(5)
train_data.shape
train_data.describe().T
# Get names of columns with missing values

cols_with_missing = [col for col in train_data.columns

                     if train_data[col].isnull().any()]

print(cols_with_missing)
#All columns are numerical

#train_data.dtypes

train_data.info()
#I change type of categorical columns in order to display correlacion easily

train_data.iloc[:,10:-1] = train_data.iloc[:,10:-1].astype("category")
f,ax = plt.subplots(figsize=(8,6))

sns.heatmap(train_data.corr(),annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()
train_data.iloc[:,:3].columns
sns.pairplot(train_data[list(train_data.iloc[:,6:9].columns)+['Cover_Type']], hue="Cover_Type");

#ns.pairplot(train_data[list(train_data.iloc[:,:3].columns)+['Cover_Type']], hue="Cover_Type", palette="colorblind", diag_kind="kde");
FG = sns.FacetGrid(train_data, hue="Cover_Type", palette="Set2", size=5) 

#                   hue_kws={"marker": ["^", "v", "*"]})

FG.map(plt.scatter, "Hillshade_9am", "Hillshade_3pm", s=50, linewidth=.5, edgecolor="white")

FG.add_legend();

plt.figure(figsize=(24,18))



for n, columns in enumerate(train_data.iloc[:,:10].columns,1):

    plt.subplot(4,3,n)

    for i in np.arange(1,8):

        sns.kdeplot(train_data[columns][(train_data['Cover_Type']==i)], label=i, shade=True)

        plt.xlabel(columns);



plt.figure(figsize=(24,9))



for i in np.arange(1,8):

    plt.subplot(2,4,i)

    plt.hist(train_data.loc[train_data['Cover_Type']==i, 'Wilderness_Area3'], range = [0,1])

    plt.xlabel(i)

        

   

    
plt.figure(figsize=(18,8))



for i in np.arange(1,8):

    plt.subplot(2,4,i)

    sns.distplot(train_data['Hillshade_9am'][(train_data['Cover_Type']==i)])

    plt.xlabel(i)
fig, axs = plt.subplots(2, 2, sharey=True, figsize = (20,15))



axs[0, 0].scatter(train_data['Hillshade_9am'], train_data['Hillshade_Noon'], color='darkblue')

axs[0, 0].set(title = 'Hillshade_9am And Hillshade_Noon', xlabel = "Hillshade_9am", ylabel = "Hillshade_Noon")



axs[0, 1].scatter(train_data['Hillshade_Noon'], train_data['Hillshade_3pm'], color='darkblue')

axs[0, 1].set(title = 'Hillshade_Noon And Hillshade_3pm', xlabel = "Hillshade_Noon", ylabel = "Hillshade_3pm")



axs[1, 0].scatter(train_data['Slope'], train_data['Hillshade_3pm'], color='darkblue')

axs[1, 0].set(title = 'Slope And Hillshade_3pm', xlabel = "Slope", ylabel = "Hillshade_3pm")



axs[1, 1].scatter(train_data['Aspect'], train_data['Hillshade_3pm'], color='darkblue')

axs[1, 1].set(title = 'Aspect And Hillshade_3pm', xlabel = "Aspect", ylabel = "Hillshade_3pm")

train_data.plot(kind='scatter', x='Slope', y='Elevation', alpha=0.5, color='maroon', figsize = (12,9))

plt.title('Slope And Elevation')

plt.xlabel("Slope")

plt.ylabel("Elevation")

plt.show()
# import modules 

from sklearn.model_selection import cross_val_predict

from sklearn import linear_model

from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score
# To prepare data

Hillshade_train_data = train_data[(train_data["Hillshade_3pm"] != 0)]

Hillshade_test_data = train_data[(train_data["Hillshade_3pm"] == 0)]

print(Hillshade_train_data.shape)

print(Hillshade_test_data.shape)

# linear regression

lr = linear_model.LinearRegression()

#lr = linear_model.LassoCV(normalize = True)

#lr = linear_model.RidgeCV(normalize = True)

scores=cross_val_score(lr,Hillshade_train_data[['Hillshade_9am', 'Hillshade_Noon', 'Aspect', 'Slope']],Hillshade_train_data["Hillshade_3pm"],cv=5,scoring='r2')

print(scores)

print(scores.mean())
lr.fit(Hillshade_train_data[['Hillshade_9am', 'Hillshade_Noon', 'Aspect', 'Slope']],Hillshade_train_data["Hillshade_3pm"])

Hillshade_test_data.loc[:,"Hillshade_3pm"] = lr.predict(Hillshade_test_data[['Hillshade_9am', 'Hillshade_Noon', 'Aspect', 'Slope']])
fig, axs = plt.subplots(2, 2, sharey=True, figsize = (20,15))



axs[0, 0].scatter(train_data['Hillshade_9am'], train_data['Hillshade_3pm'], color='darkblue')

axs[0, 0].scatter(Hillshade_test_data['Hillshade_9am'], Hillshade_test_data['Hillshade_3pm'], color='maroon')

axs[0, 0].set(title = 'Hillshade_9am And Hillshade_3pm', xlabel = "Hillshade_9am", ylabel = "Hillshade_3pm")



axs[0, 1].scatter(train_data['Hillshade_Noon'], train_data['Hillshade_3pm'], color='darkblue')

axs[0, 1].scatter(Hillshade_test_data['Hillshade_Noon'], Hillshade_test_data['Hillshade_3pm'], color='maroon')

axs[0, 1].set(title = 'Hillshade_Noon And Hillshade_3pm', xlabel = "Hillshade_Noon", ylabel = "Hillshade_3pm")



axs[1, 0].scatter(train_data['Aspect'], train_data['Hillshade_3pm'], color='darkblue')

axs[1, 0].scatter(Hillshade_test_data['Aspect'], Hillshade_test_data['Hillshade_3pm'], color='maroon')

axs[1, 0].set(title = 'Aspect And Hillshade_3pm', xlabel = "Aspect", ylabel = "Hillshade_3pm")



axs[1, 1].scatter(train_data['Slope'], train_data['Hillshade_3pm'], color='darkblue')

axs[1, 1].scatter(Hillshade_test_data['Slope'], Hillshade_test_data['Hillshade_3pm'], color='maroon')

axs[1, 1].set(title = 'Slope And Hillshade_3pm', xlabel = "Slope", ylabel = "Hillshade_3pm")



plt.show()
# Se reemplazan los zero de "Hillshade_3pm" en train_data 

train_data.loc[(train_data["Hillshade_3pm"] == 0),"Hillshade_3pm"]=lr.predict(Hillshade_test_data[['Hillshade_9am', 'Hillshade_Noon', 'Aspect', 'Slope']])
fig, axs = plt.subplots(2, 2, sharey=True, figsize = (20,15))



axs[0, 0].scatter(train_data['Hillshade_9am'], train_data['Hillshade_3pm'], color='darkblue')

axs[0, 0].scatter(Hillshade_test_data['Hillshade_9am'], Hillshade_test_data['Hillshade_3pm'], color='maroon')

axs[0, 0].set(title = 'Hillshade_9am And Hillshade_3pm', xlabel = "Hillshade_9am", ylabel = "Hillshade_3pm")



axs[0, 1].scatter(train_data['Hillshade_Noon'], train_data['Hillshade_3pm'], color='darkblue')

axs[0, 1].scatter(Hillshade_test_data['Hillshade_Noon'], Hillshade_test_data['Hillshade_3pm'], color='maroon')

axs[0, 1].set(title = 'Hillshade_Noon And Hillshade_3pm', xlabel = "Hillshade_Noon", ylabel = "Hillshade_3pm")



axs[1, 0].scatter(train_data['Aspect'], train_data['Hillshade_3pm'], color='darkblue')

axs[1, 0].scatter(Hillshade_test_data['Aspect'], Hillshade_test_data['Hillshade_3pm'], color='maroon')

axs[1, 0].set(title = 'Aspect And Hillshade_3pm', xlabel = "Aspect", ylabel = "Hillshade_3pm")



axs[1, 1].scatter(train_data['Slope'], train_data['Hillshade_3pm'], color='darkblue')

axs[1, 1].scatter(Hillshade_test_data['Slope'], Hillshade_test_data['Hillshade_3pm'], color='maroon')

axs[1, 1].set(title = 'Slope And Hillshade_3pm', xlabel = "Slope", ylabel = "Hillshade_3pm")



plt.show()
train_data.plot(kind='scatter', x='Aspect', y='Hillshade_9am', alpha=0.5, color='maroon', figsize = (12,9))

plt.title('Aspect And Hillshade_9am')

plt.xlabel("Aspect")

plt.ylabel("Hillshade_9am")

plt.show()
train_data.plot(kind='scatter', y='Hillshade_Noon', x='Slope', alpha=0.5, color='darkblue', figsize = (12,9))

plt.title('Hillshade_Noon And Slope')

plt.ylabel("Hillshade_Noon")

plt.xlabel("Slope")

plt.show()
import plotly.graph_objs as go

from plotly.offline import iplot





trace1 = go.Box(

    y=train_data["Vertical_Distance_To_Hydrology"],

    name = 'Vertical Distance',

    marker = dict(color = 'rgb(0,145,119)')

)

trace2 = go.Box(

    y=train_data["Horizontal_Distance_To_Hydrology"],

    name = 'Horizontal Distance',

    marker = dict(color = 'rgb(5, 79, 174)')

)



data = [trace1, trace2]

layout = dict(autosize=False, width=700,height=500, title='Distance To Hydrology', paper_bgcolor='rgb(243, 243, 243)', 

              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))

fig = dict(data=data, layout=layout)

iplot(fig)

f,ax=plt.subplots(1,2,figsize=(15,7))

train_data.Vertical_Distance_To_Hydrology.plot.hist(ax=ax[0],bins=30,edgecolor='black',color='crimson')

ax[0].set_title('Vertical Distance To Hydrology')

x1=list(range(-150,350,50))

ax[0].set_xticks(x1)

train_data.Horizontal_Distance_To_Hydrology.plot.hist(ax=ax[1],bins=30,edgecolor='black',color='darkmagenta')

ax[1].set_title('Horizontal Distance To Hydrology')

x2=list(range(0,1000,100))

ax[1].set_xticks(x2)

plt.show()
import plotly.graph_objs as go

from plotly.offline import iplot



trace1 = go.Box(

    y=train_data["Horizontal_Distance_To_Roadways"],

    name = 'Distance_To_Roadways',

    marker = dict(color = 'rgb(0,145,119)')

)

trace2 = go.Box(

    y=train_data["Horizontal_Distance_To_Fire_Points"],

    name = 'Distance_To_Fire_Points',

    marker = dict(color = 'rgb(5, 79, 174)')

)



data = [trace1, trace2]

layout = dict(autosize=False, width=700,height=500, title='Other Distances', paper_bgcolor='rgb(243, 243, 243)', 

              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))

fig = dict(data=data, layout=layout)

iplot(fig)
f,ax=plt.subplots(1,2,figsize=(15,7))

train_data.Horizontal_Distance_To_Roadways.plot.hist(ax=ax[0],bins=30,edgecolor='black',color='crimson')

ax[0].set_title('Horizontal_Distance_To_Roadways')

x1=list(range(0,7000,500))

ax[0].set_xticks(x1)



train_data.Horizontal_Distance_To_Fire_Points.plot.hist(ax=ax[1],bins=30,edgecolor='black',color='darkmagenta')

ax[1].set_title('Horizontal_Distance_To_Fire_Points')

x2=list(range(0,7000,500))

ax[1].set_xticks(x2)

plt.show()
print(train_data.shape)

# Suppression of outliers

# values 0 in Hillshade_3pm and in ¿Aspect?

#Type 7 (0), Type 8 (1), Type 15 (0) and Type 25 (1) have either no or too few values.

# Type 9 (10), type 28 (9), type 36 (10),

# type 21 (16) type 27 (15), type 34 (22), type 37 (34)



#reduce_train_data=train_data[(train_data["Horizontal_Distance_To_Hydrology"]<750) 

#                             & (train_data["Vertical_Distance_To_Hydrology"]<250)

#                             & (train_data["Horizontal_Distance_To_Roadways"] < 5500)

#                             & (train_data["Horizontal_Distance_To_Fire_Points"]<4500)

#                             & (train_data['Hillshade_3pm']!=0)

#                            ]



#reduce_train_data=train_data[(train_data['Hillshade_3pm']!=0)

#                            ]



#print(reduce_train_data.shape)
train_data.Soil_Type21.value_counts()
train_data[train_data["Soil_Type21"]==1].Cover_Type.value_counts()
train_data.columns
# Create target object and call it y

y = train_data.Cover_Type

# Create X with all columns as a first asumption

#X = train_data.drop(['Cover_Type'], axis=1)

#dropped_columns=['Hillshade_3pm','Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type15','Soil_Type21','Soil_Type25','Soil_Type27','Soil_Type28','Soil_Type34','Soil_Type36','Soil_Type37']

dropped_columns=['Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type15','Soil_Type25', 'Soil_Type28','Soil_Type36', 'Hillshade_9am', 'Hillshade_Noon','Hillshade_3pm', 'Vertical_Distance_To_Hydrology']

#dropped_columns=['Soil_Type7','Soil_Type8','Soil_Type15','Soil_Type25']

X = train_data.drop(dropped_columns+['Cover_Type'], axis=1)
f,ax=plt.subplots(1,2,figsize=(15,7))

train_data.Hillshade_3pm.plot.hist(ax=ax[0],bins=30,edgecolor='black',color='crimson')

ax[0].set_title('Hillshade_3pm')

x1=list(range(0,300,20))

ax[0].set_xticks(x1)

train_data.Aspect.plot.hist(ax=ax[1],bins=30,edgecolor='black',color='darkmagenta')

ax[1].set_title('Aspect')

x2=list(range(0,100,20))

ax[1].set_xticks(x2)

plt.show()
import plotly.express as px

cover_type = train_data["Cover_Type"].value_counts()

df_cover_type = pd.DataFrame({'CoverType': cover_type.index, 'Total':cover_type.values})



fig = px.bar(df_cover_type, x='CoverType', y='Total', height=400, width=650)

fig.show()
import pandas_profiling as pp

#report = pp.ProfileReport(train_data)

#report.to_file("report.html")



#report
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold

kf = KFold(n_splits=5, shuffle = True ,random_state=1)



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=2)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, roc_curve,confusion_matrix



# First test with all features and random forest

# Define the model. Set random_state to 1

rf_scores={}

#for n in [100]:

#    rf_model = RandomForestClassifier(n_estimators=n, random_state=1)

#    rf_model.fit(train_X, train_y)

#    rf_val_predictions = rf_model.predict(val_X)

#    rf_acc = accuracy_score(val_y, rf_val_predictions)

    #print (rf_acc)

#    rf_scores[n] = rf_acc



#print(rf_scores)



#rf_mat = confusion_matrix(val_y, rf_val_predictions)

#print (rf_mat)





#kf = KFold(n_splits=5, shuffle = True ,random_state=1)

#precisions = cross_val_score(rf_model, train_X, train_y, cv=kf, n_jobs=1, scoring = 'precision',verbose = 0)

#recalls = cross_val_score(rf_model, train_X, train_y, cv=kf, n_jobs=1, scoring = 'recall',verbose = 0)



#print('recall = %f, precision = %f' %(recalls.mean(), precisions.mean()))

# second test with all features and Gradient Boosting

# Define the model. Set random_state to 1

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, confusion_matrix



#for n in [800,1000,1200]:

#    gb_model = XGBClassifier(n_estimators=n, random_state=1)

#    gb_model.fit(train_X, train_y)

#    gb_val_predictions = gb_model.predict(val_X)



#    gb_acc = accuracy_score(val_y, gb_val_predictions)

#    print (gb_acc)



#gb_model.fit(train_X, train_y,

#             early_stopping_rounds=5, 

#             eval_set=[(val_X, val_y)],

#             verbose=False)





#print('')



#gb_mat = confusion_matrix(val_y, gb_val_predictions)

#print (gb_mat)
#third test



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier



def evaluar_rendimiento(modelo, nombre):

    s = cross_val_score(modelo, X, y, cv=kf, n_jobs=-1)

    print("Rendimiento de {}:\t{:0.3} ± {:0.3}".format( \

        nombre, s.mean().round(3), s.std().round(3)))

    

    

dt = DecisionTreeClassifier(class_weight='balanced')



#evaluar_rendimiento(dt,"Árbol de decisión")

bdt = BaggingClassifier(DecisionTreeClassifier())

rf = RandomForestClassifier(class_weight='balanced')

et = ExtraTreesClassifier(class_weight='balanced')



#evaluar_rendimiento(dt,  "Árbol de decisión")

#evaluar_rendimiento(bdt, "Bagging AD")

#evaluar_rendimiento(rf,  "Random Forest")

#evaluar_rendimiento(et,  "Extra Trees")
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_trees = {'n_estimators': [100, 150, 200, 250, 300], 

               #'max_features': ['auto', 'sqrt'], 

               'max_features': [32, 36, 40, 42],

               'max_depth': [75, 100, 150, 200], 

               'min_samples_split' : [2],

               'min_samples_leaf':[1]}



rf = ExtraTreesClassifier(class_weight='balanced')

kf = StratifiedKFold(n_splits=4, shuffle=True)



#grid_search_rf = GridSearchCV(rf, param_grid=param_trees, cv=kf, verbose=1, n_jobs=-1)

#grid_search_et = RandomizedSearchCV(estimator = rf, 

#                                    param_distributions = param_trees, 

#                                    n_iter = 10, 

#                                    cv = kf, 

#                                    verbose=1, 

#                                    random_state=2, 

#                                    n_jobs = -1, 

#                                    scoring = 'accuracy')



#grid_search_rf.fit(X, y)

#grid_search_rf.best_estimator_
#grid_search_rf.best_score_
from sklearn.model_selection import cross_val_score



#rf_model = ExtraTreesClassifier(n_estimators=100, max_features=48, max_depth=100, min_samples_leaf =1, random_state=2) # 0.8758597883597884 / 

rf_model = ExtraTreesClassifier(n_estimators=250, max_features=32, max_depth=150, min_samples_leaf =1, random_state=2) # 0.8766534391534392 ==> 0.78071

#rf_model = ExtraTreesClassifier(n_estimators=300, max_features=40, max_depth=100, min_samples_leaf =1, random_state=2) # 0.875462962962963

#rf_model = ExtraTreesClassifier(n_estimators=200, max_features=30, max_depth=200, min_samples_leaf =1, random_state=2) #0.8752645502645503

#rf_model = ExtraTreesClassifier(n_estimators=100, max_features=44, max_depth=75, min_samples_leaf =1, random_state=2) # 0.8767195767195767 => 0.77934



#rf_model = RandomForestClassifier(n_estimators = 719, max_features = 0.3, max_depth = 464, min_samples_split = 2,min_samples_leaf = 1,bootstrap = False, random_state=2) # 0.8712962962962962

#rf_model = ExtraTreesClassifier(n_estimators = 719, max_features = 0.3, max_depth = 464, min_samples_split = 2,min_samples_leaf = 1,bootstrap = False, random_state=2) # 0.8732804232804232



#rf_model.fit(train_X, train_y)

#rf_val_predictions = rf_model.predict(val_X)

#rf_acc = accuracy_score(val_y, rf_val_predictions)

#print (rf_acc)



scores = cross_val_score(rf_model, X, y, cv=kf,scoring='accuracy')

print (scores)



print("Average accuracy:")

print(scores.mean())
model = ExtraTreesClassifier(n_estimators=250, max_features=32, max_depth=150, min_samples_leaf =1, random_state=2)

model.fit(X, y)

test_data = pd.read_csv("../input/learn-together/test.csv",index_col='Id')

test_data.sample(5)
print(test_data.loc[(test_data["Hillshade_3pm"] == 0)].shape)

# Se reemplazan los zero de "Hillshade_3pm" en test_data 

test_data.loc[(test_data["Hillshade_3pm"] == 0),"Hillshade_3pm"]=lr.predict(test_data.loc[(test_data["Hillshade_3pm"] == 0),['Hillshade_9am', 'Hillshade_Noon', 'Aspect', 'Slope']])



print(test_data.loc[(test_data["Hillshade_3pm"] == 0)].shape)
test_X = test_data.drop(dropped_columns, axis=1)

test_X.shape
# make predictions which we will submit. 

test_preds = model.predict(test_X)



print(test_preds[0:10])



output = pd.DataFrame({'ID': test_data.index,

                       'Cover_Type': test_preds})

output.to_csv('submission.csv', index=False)



#print(pd.read_csv('submission.csv'))