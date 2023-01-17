from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
data.head()
data.info()
data.describe()
%matplotlib inline
import matplotlib.pyplot as plt
data.hist(bins=50,figsize=(12,9))
plt.show()


corr_matrix= data.corr()
corr_matrix['Response'].sort_values(ascending=False)
import seaborn as sns
f, ax = plt.subplots(figsize=(12,12))
sns.heatmap(data.corr(),annot=True, linewidths=.5, ax=ax)
plt.show()


import matplotlib.pyplot as plt

labels ='Not-responed', 'Responed'
sizes = [len(data[data['Response']==0]),len(data[data['Response']==1])]
explode = (0, 0.04) 

fig1, ax1 = plt.subplots(figsize=(8,8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, colors=('r','yellow'), startangle=90)
ax1.set_title("Response Events")
ax1.axis('equal') 

plt.show()


cat_2=['Gender','Previously_Insured','Vehicle_Damage']

types=[['Women','Men'],['No','Yes'],['No','Yes']]
for i,c in enumerate(cat_2):
    alive = data[data['Response']==0]
    died= data[data['Response']==1]
    plt.figure(figsize=(8,5))
    bar1=plt.bar(np.arange(len(data[c].unique())), alive.groupby(c).count()['Age'], width=0.1, color='orange', align='center', label="Not responed")
    bar2= plt.bar(np.arange(len(data[c].unique()))+0.1, died.groupby(c).count()['Age'], width=0.1, color='green', align='center', label="responed")
    plt.title(c)
    #plt.ylim(0,160)
    plt.xticks([0,1], types[i])
    plt.grid()
    plt.legend()

    hights_odd=[]
    hights_even=[]
    for i,rect in enumerate (bar1 + bar2):
        height = rect.get_height()
        if (i+1)%2==0:
            hights_even.append(height)
        if (i+1)%2!=0:
            hights_odd.append(height)

    for i,rect in enumerate (bar1 + bar2):
        height = rect.get_height()
        if (i+1)%2==0:
            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%s' % str(round((height/sum(hights_even)*100),2))+"%", ha='center', va='bottom')
        if (i+1)%2!=0:
            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%s' % str(round((height/sum(hights_odd))*100,2))+"%", ha='center', va='bottom')
plt.figure(figsize=(10,7))
plt.title('Region Code')
plt.grid()
maxx=0
high=[]
xs=[]
for i in sorted(data['Region_Code'].unique()):
    bar= plt.bar(i,len(data[data['Region_Code']==i]))    

import matplotlib.pyplot as plt

labels ='region 28', 'all but not 28'
sizes = [len(data[data['Region_Code']==28]),len(data[data['Region_Code']!=28])]
explode = (0, 0.04) 

fig1, ax1 = plt.subplots(figsize=(8,8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, colors=('b','c'), startangle=90)
ax1.set_title("Response Events")
ax1.axis('equal') 

plt.show()
data['Vehicle_Age'].unique()
time_1= data[data['Vehicle_Age']=='< 1 Year']
time_2= data[data['Vehicle_Age']=='1-2 Year']
time_3= data[data['Vehicle_Age']=='> 2 Years']

explode = (0, 0.05)
labels = 'Not responsive', 'Responsive'

types= [time_1,time_2,time_3,]
fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(13,7))
ax= (ax1, ax2,ax3)
fig.suptitle('Vehicle Age virsus Reponses',fontsize=20)
titles= ['Less than a year', '1-2 years', 'More than two years']



for ax, typ,title in zip(ax,types,titles ):
    
    sizes = [len(typ[typ['Response']==0]),len(typ[typ['Response']==1])]
    wedges, texts,autopct = ax.pie(sizes, autopct='%1.1f%%', explode=explode,colors=['r','y'], labels=labels)
    ax.set_title(title)
    
    ax.axis('equal') 
plt.show()
plt.figure(figsize=(10,7))
plt.title('Policy Sales Channel')
plt.grid()
maxx=0
for i in sorted(data['Policy_Sales_Channel'].unique()):
    bar= plt.bar(i,len(data[data['Policy_Sales_Channel']==i]))

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

types= [data[data['Response']==0]['Age'], data[data['Response']==1]['Age']]
titles= [ 'Age Distribution for People Who Doesnt Responded with ', 'Age Distribution for People Who Responded with ']
colors=['r','blue']
#age= data['Age']

for age, tit,color in zip(types, titles,colors):
    mu, std = norm.fit(age)
    plt.figure(figsize=(12,7))
    plt.hist(age, bins=25, density=True, alpha=0.6, color=color)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    tit +="mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(tit)
    plt.grid()
    plt.show()                

plt.figure(figsize=(10,8))
plt.xticks([1,2], ['Responeded', 'Not'])
plt.boxplot(types)
plt.title('Boxplot for Age cat')
plt.grid()
plt.show()

v_1= data[data['Vintage']<100]
v_2= data[data['Vintage']>100][data['Vintage']<200]
v_3= data[data['Vintage']>200]

explode = (0, 0.05)
labels = 'Not responsive', 'Responsive'

types= [v_1,v_2,v_3]
fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(13,7))
ax= (ax1, ax2,ax3)
fig.suptitle('Vehicle Age vs Reponses',fontsize=20)
titles= ['Less than a year', '1-2 years', 'More than two years']



for ax, typ,title in zip(ax,types,titles ):
    
    sizes = [len(typ[typ['Response']==0]),len(typ[typ['Response']==1])]
    wedges, texts,autopct = ax.pie(sizes, autopct='%1.1f%%', explode=explode,colors=['r','y'], labels=labels)
    ax.set_title(title)
    
    ax.axis('equal') 
plt.show()
plt.figure(figsize=(8,6))
plt.bar([0,1,2], [len(v_1),len(v_2),len(v_3)], color='g')
plt.xticks([0,1,2], ['Group1','Group2', 'Group3'])
plt.title("Vintage Groups Data Distribution")
plt.grid()
min(data['Annual_Premium'])
max(data['Annual_Premium'])
#Convert to US Dollar
data['Annual_Premium_$']= data['Annual_Premium']*0.014
print(min(data['Annual_Premium_$']))
print(max(data['Annual_Premium_$']))
print(np.median(data['Annual_Premium_$']))
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

ap= data['Annual_Premium_$']
#age= data['Age']

mu, std = norm.fit(age)
plt.figure(figsize=(10,7))
plt.hist(ap, bins=25, density=True, alpha=0.6, color='r')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
tit ="mu = %.2f,  std = %.2f" % (mu, std)
plt.title("Annual Premium--"+ tit)
plt.grid()
plt.show() 
a_1= data[data['Annual_Premium_$']<=442]
a_2= data[data['Annual_Premium_$']>442]

explode = (0, 0.05)
labels = 'Not responsive', 'Responsive'

types= [a_1,a_2]
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,7))
ax= (ax1, ax2)
fig.suptitle('Annual Premium ',fontsize=20)
titles= ['Less than mean value', 'More than mean value']



for ax, typ,title in zip(ax,types,titles ):
    
    sizes = [len(typ[typ['Response']==0]),len(typ[typ['Response']==1])]
    wedges, texts,autopct = ax.pie(sizes, autopct='%1.1f%%', explode=explode,colors=['r','y'], labels=labels)
    ax.set_title(title)
    
    ax.axis('equal') 
plt.show()
plt.figure(figsize=(8,6))
plt.bar([0,1], [len(a_1),len(a_2)], color='g')
plt.xticks([0,1], ['Group1','Group2'])
plt.title("Vintage Groups Data Distribution")
plt.grid()
data['Gender']=data['Gender'].astype('category').cat.codes
data['Vehicle_Age']= [0 if data['Vehicle_Age'][i]=='< 1 Year' else 1 if data['Vehicle_Age'][i]=='1-2 Year' else 2 for i in range(len(data['Vehicle_Age']))]
#data['Vehicle_Age'] = data['Gender'].astype('category')
data['Vehicle_Damage']=data['Vehicle_Damage'].astype('category').cat.codes
data['Region_Code']= data['Region_Code'].astype(int)
#data['Policy_Sales_Channel']= data['Policy_Sales_Channel'].astype('category')
data.columns
features=[ 'Gender', 'Age','Region_Code',
       'Previously_Insured', 'Vehicle_Age','Vehicle_Damage', 'Annual_Premium',
       'Policy_Sales_Channel', 'Vintage', 'Response']

num=[ 'Age','Annual_Premium','Vintage']

train_prep=data[features]
from category_encoders import TargetEncoder

encoder = TargetEncoder()
train_prep['Region_Code'] = encoder.fit_transform(train_prep['Region_Code'], train_prep['Response'])
train_prep['Policy_Sales_Channel'] = encoder.fit_transform(train_prep['Policy_Sales_Channel'], train_prep['Response'])

#train_prep['Region_Code'] = train_prep['Region_Code'].astype('category',copy=False)
#train_prep= pd.get_dummies(train_prep)

from sklearn.preprocessing import StandardScaler
std=StandardScaler()
train_prep[num]= std.fit_transform(train_prep[num])

train_prep
from sklearn.model_selection import StratifiedShuffleSplit 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, valid_index in split.split(train_prep, train_prep["Response"]):
    train = train_prep.loc[train_index]
    valid = train_prep.loc[valid_index]
from imblearn.over_sampling import SMOTE

# Resample the minority class. You can change the strategy to 'auto' if you are not sure.
sm = SMOTE(sampling_strategy='minority', random_state=7)

# Fit the model to generate the data.
oversampled_trainX, oversampled_trainY = sm.fit_sample(train.drop('Response', axis=1), train['Response'])
oversampled_train = pd.concat([oversampled_trainX, oversampled_trainY], axis=1)
oversampled_train
y_train= oversampled_train['Response']
y_valid= valid['Response']

X_train= oversampled_train.drop('Response', axis=1)
X_valid= valid.drop('Response', axis=1)

X_train.index = np.arange(len(X_train))
X_valid.index = np.arange(len(X_valid))

y_train.index = np.arange(len(y_train))
y_valid.index = np.arange(len(y_valid))
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X_train, y_train)

tree_preds = tree_clf.predict(X_valid)
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
print("Acc:",accuracy_score(y_valid, tree_preds))

print("Precision:",precision_score(y_valid, tree_preds))

print("Recall:",recall_score(y_valid, tree_preds))

print('f1-score', f1_score(y_valid, tree_preds))

print('ROC score', roc_auc_score(y_valid, tree_preds))
import os
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
from graphviz import Source
from sklearn.tree import export_graphviz

export_graphviz(
        tree_clf,
        out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),
        feature_names=X_train.columns,
        class_names=['not resp', 'resp'],
        rounded=True,
        filled=True
    )

Source.from_file(os.path.join(IMAGES_PATH, "iris_tree.dot"))
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


kf = KFold(n_splits=2)

max_features = ['auto', 'sqrt','log2', None]

rf_Model = RandomForestClassifier()

rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = {'max_features':max_features}, cv = kf,  scoring='accuracy',n_jobs=-1, verbose=4)

rf_Grid.fit(X_train, y_train)
grid_results = pd.concat([pd.DataFrame(rf_Grid.cv_results_["params"]),pd.DataFrame(rf_Grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
grid_results
rf_Grid.best_params_
scores_test = rf_Grid.cv_results_['mean_test_score']
#scores = np.array(scores).reshape(len(Cs), len(n_estimators))
plt.figure(figsize=(10,6))
plt.plot([0,1,2,3], scores_test, label="Testing Error")
plt.xticks([0,1,2,3], ['auto', 'sqrt', 'log2', 'None'])
plt.legend()
plt.xlabel('n_estimators')
plt.ylabel('Mean score')
plt.grid()
plt.show()
min_samples_leaf=[1,2,4, 6]
    
rf_leaf_Model = RandomForestClassifier()

rf_leaf_Grid = GridSearchCV(estimator = rf_leaf_Model, param_grid = {'min_samples_leaf':min_samples_leaf}, cv = kf, verbose=5, n_jobs = -1)

rf_leaf_Grid.fit(X_train, y_train)

grid_leaf_results = pd.concat([pd.DataFrame(rf_leaf_Grid.cv_results_["params"]),pd.DataFrame(rf_leaf_Grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
grid_leaf_results
rf_leaf_Grid.best_params_
max_depth = [None,2,4,6]

rf_depth_Model = RandomForestClassifier()

rf_dep_Grid = GridSearchCV(estimator = rf_depth_Model, param_grid = {'max_depth':max_depth}, cv = kf, verbose=2, n_jobs = -1)

rf_dep_Grid.fit(X_train, y_train)

grid_results = pd.concat([pd.DataFrame(rf_dep_Grid.cv_results_["params"]),pd.DataFrame(rf_dep_Grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
grid_results.head()
rf_dep_Grid.best_params_
min_samples_split = [2,5,7]

rf_mss_Model = RandomForestClassifier()

rf_mss_Grid = GridSearchCV(estimator = rf_mss_Model, param_grid = {'min_samples_split':min_samples_split}, cv = kf, verbose=2, n_jobs = -1)

rf_mss_Grid.fit(X_train, y_train)

grid_results = pd.concat([pd.DataFrame(rf_mss_Grid.cv_results_["params"]),pd.DataFrame(rf_mss_Grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
grid_results.head()
rf_mss_Grid.best_params_
rnd_clf = RandomForestClassifier( max_features=None,max_depth= None,
                                 min_samples_leaf=1,min_samples_split=2, random_state=42)

rnd_clf.fit(X_train, y_train)
rnf_preds= rnd_clf.predict(X_valid)
print("Acc:",accuracy_score(y_valid, rnf_preds))

print("Precision:",precision_score(y_valid, rnf_preds))

print("Recall:",recall_score(y_valid, rnf_preds))

print('f1-score', f1_score(y_valid, rnf_preds))

print('ROC score', roc_auc_score(y_valid, rnf_preds))
from sklearn.metrics import roc_curve,auc
y_score = rnd_clf.predict_proba(X_valid)[:,1]
fpr, tpr, _ = roc_curve(y_valid,y_score)
import matplotlib.pyplot as plt

plt.title('Random Forest ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr))
import pickle
filename = 'rf_clf.sav'
pickle.dump(rnd_clf, open(filename, 'wb'))

filename = 'rf_clf.sav'
rf_load = pickle.load(open(filename, 'rb'))
from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier()
lr= [0.01, 0.1,0.5, 1.0]
search_grid={'learning_rate':lr}
search=GridSearchCV(estimator=ada,param_grid=search_grid,scoring='accuracy',n_jobs=1,cv=kf,verbose=2)
search.fit(X_train, y_train)
grid_results = pd.concat([pd.DataFrame(search.cv_results_["params"]),
                          pd.DataFrame(search.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
grid_contour = grid_results.groupby(['learning_rate']).mean()
grid_contour
print(search.best_score_)
print(search.best_params_)
search.best_params_
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=300, learning_rate=1, random_state=42)
ada_clf.fit(X_train, y_train)
ada_pred= ada_clf.predict(X_valid)
print("Acc:",accuracy_score(y_valid, ada_pred))

print("Precision:",precision_score(y_valid, ada_pred))

print("Recall:",recall_score(y_valid, ada_pred))

print('f1-score', f1_score(y_valid, ada_pred))

print('ROC score', roc_auc_score(y_valid, ada_pred))
from sklearn.metrics import roc_curve,auc
y_score = ada_clf.predict_proba(X_valid)[:,1]
fpr, tpr, _ = roc_curve(y_valid,y_score)
import matplotlib.pyplot as plt

plt.title('AdaBoost ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr))
filename = 'ada_clf.sav'
pickle.dump(ada_clf, open(filename, 'wb'))

filename = 'ada_clf.sav'
rf_load = pickle.load(open(filename, 'rb'))
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state = 42)
gb.fit(X_train, y_train)
gbrt_pred= gb.predict(X_valid)
print("Acc:",accuracy_score(y_valid, gbrt_pred))

print("Precision:",precision_score(y_valid, gbrt_pred))

print("Recall:",recall_score(y_valid, gbrt_pred))

print('f1-score', f1_score(y_valid, gbrt_pred))

print('ROC score', roc_auc_score(y_valid, gbrt_pred))
y_score = gb.predict_proba(X_valid)[:,1]
fpr, tpr, _ = roc_curve(y_valid,y_score)

plt.title('Gadient Boosting ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr))
filename = 'gb_clf.sav'
pickle.dump(gb, open(filename, 'wb'))

from xgboost import XGBClassifier

param_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        }
xgboost_model = XGBClassifier()

xgboost_search= GridSearchCV(estimator = xgboost_model, param_grid = param_grid, cv = 2, verbose=10, n_jobs = -1)

xgboost_search.fit(X_train, y_train)

grid_results = pd.concat([pd.DataFrame(xgboost_search.cv_results_["params"]),
                          pd.DataFrame(xgboost_search.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
grid_contour = grid_results.groupby(['gamma','min_child_weight']).mean()
grid_contour
xgboost_search.best_params_
xgboost_clf = XGBClassifier(gamma= 2, min_child_weight=1)
xgboost_clf.fit(X_train, y_train)
xgb_pred= xgboost_clf.predict(X_valid)
print("Acc:",accuracy_score(y_valid, xgb_pred))

print("Precision:",precision_score(y_valid, xgb_pred))

print("Recall:",recall_score(y_valid, xgb_pred))

print('f1-score', f1_score(y_valid, xgb_pred))

print('ROC score', roc_auc_score(y_valid, xgb_pred))
y_score = xgboost_clf.predict_proba(X_valid)[:,1]
fpr, tpr, _ = roc_curve(y_valid,y_score)

plt.title('XGBoost ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr))
filename = 'xgb_clf.sav'
pickle.dump(xgb_pred, open(filename, 'wb'))


proba_tree,proba_rnd, proba_ada, proba_gb, proba_xg = tree_clf.predict_proba(X_valid)[:,1], rnd_clf.predict_proba(X_valid)[:,1],ada_clf.predict_proba(X_valid)[:,1],gb.predict_proba(X_valid)[:,1],xgboost_clf.predict_proba(X_valid)[:,1]

preds= [proba_tree,proba_rnd, proba_ada, proba_gb, proba_xg]

labels= ['DT','Random Forest', 'AdaBoost',"Gradient Boosting",'XGBoost']
plt.figure(figsize=(10,8))

for pred, label in zip(preds,labels):
    fpr, tpr, thresholds = roc_curve(y_valid, pred)
    plt.plot(fpr, tpr, linewidth=2, label=label)
plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
plt.axis([0, 1, 0, 1])                                    
plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) 
plt.ylabel('True Positive Rate (Recall)', fontsize=16)    
plt.grid(True)  
plt.legend()
result = []

results = pd.DataFrame(columns= ["Models","Accuracy"])

for model in [tree_clf, rnd_clf, ada_clf, gb, xgboost_clf]:
    names = model.__class__.__name__
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)    
    result = pd.DataFrame([[names, accuracy*100]], columns= ["Models","Accuracy"])
    results = results.append(result)
    
    
sns.barplot(x= 'Accuracy', y = 'Models', data=results, color="r")
plt.xlabel('Accuracy %')
plt.title('Accuracy Ratios of Models'); 

test = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')
test.head()
test['Gender']=test['Gender'].astype('category').cat.codes
test['Vehicle_Age']= [0 if test['Vehicle_Age'][i]=='< 1 Year' else 1 if test['Vehicle_Age'][i]=='1-2 Year' else 2 for i in range(len(test['Vehicle_Age']))]
#data['Vehicle_Age'] = data['Gender'].astype('category')
test['Vehicle_Damage']=test['Vehicle_Damage'].astype('category').cat.codes
test['Region_Code']= test['Region_Code'].astype(int)
features=[ 'Gender', 'Age','Region_Code',
       'Previously_Insured', 'Vehicle_Age','Vehicle_Damage', 'Annual_Premium',
       'Policy_Sales_Channel', 'Vintage']

num=[ 'Age','Annual_Premium','Vintage']

test_prep=test[features]

std=StandardScaler()
test_prep[num]= std.fit_transform(test_prep[num])
test_prep.head()

models_preds={}

models= [tree_clf, rnd_clf, ada_clf, gb, xgboost_clf]

for model in models:
    name = model.__class__.__name__
    
    models_preds[name]= model.predict(test_prep)
models_preds.keys()
sub_rnd= pd.concat([pd.DataFrame(test['id']), pd.DataFrame(models_preds['RandomForestClassifier'])] ,axis=1)
sub_rnd.columns=['id', 'Response']
sub_rnd.to_csv(r'sub_rnd.csv')

sub_ada= pd.concat([pd.DataFrame(test['id']), pd.DataFrame(models_preds['AdaBoostClassifier'])] ,axis=1)
sub_ada.columns=['id', 'Response']
sub_ada.to_csv(r'sub_ada.csv')

sub_gb= pd.concat([pd.DataFrame(test['id']), pd.DataFrame(models_preds['GradientBoostingClassifier'])] ,axis=1)
sub_gb.columns=['id', 'Response']
sub_gb.to_csv(r'sub_gb.csv')

sub_xgb= pd.concat([pd.DataFrame(test['id']), pd.DataFrame(models_preds['XGBClassifier'])] ,axis=1)
sub_xgb.columns=['id', 'Response']
sub_xgb.to_csv(r'sub_xgb.csv')