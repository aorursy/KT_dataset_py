# Python libraries

# Classic,data manipulation and linear algebra

from datetime import datetime

from scipy import interp

import pandas as pd

import numpy as np

import itertools



# Plots

import shap

shap.initjs()

%matplotlib inline

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib import rcParams



# Data processing, metrics and modeling

from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, auc

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

import catboost as cb



# Filter werning

import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 500)



#Timer

def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('Time taken for Modeling: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
#Dataset

data = pd.read_csv('../input/heart-disease-uci/heart.csv')
display(data.head(), data.describe(), data.shape)
colors = ['darkturquoise', 'darkorange']

plt.style.use('dark_background')

plt.rcParams['figure.figsize']=(15,8)



ax = sns.countplot(x='target', data=data, palette=colors, alpha=0.9, edgecolor=('white'), linewidth=2)

ax.set_ylabel('count', fontsize=12)

ax.set_xlabel('target', fontsize=12)

ax.grid(b=True, which='major', color='grey', linewidth=0.2)

plt.title('Target count', fontsize=18)

plt.show()



target_0 = len(data[data.target == 0])

target_1 = len(data[data.target == 1])

print("Percentage Haven't Heart Disease: {:.2f}%".format((target_0 / (len(data.target))*100)))

print("Percentage Have Heart Disease: {:.2f}%".format((target_1 / (len(data.target))*100)))
# Correlation matrix 

f, (ax1, ax2) = plt.subplots(1,2,figsize =(15, 8))

corr = data.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

heatmapkws = dict(linewidths=0.1) 

sns.heatmap((data[data['target'] ==1]).corr(), vmax = .8, square=True, ax = ax2, cmap = 'YlGnBu', mask=mask, **heatmapkws);

ax1.set_title('Disease', fontsize=18)

sns.heatmap((data[data['target'] ==0]).corr(), vmax = .8, square=True, ax = ax1, cmap = 'afmhot', mask=mask,**heatmapkws);

ax2.set_title('Healthy', fontsize=18)

plt.show()
f,ax=plt.subplots(3,2,figsize=(12,12))

f.delaxes(ax[2,1])



for i,feature in enumerate(['age','thalach','chol','trestbps','oldpeak']):

    sns.distplot(data[feature], ax=ax[i//2,i%2], kde_kws={"color":"white"}, hist=False )



    # Get the two lines from the ax[i//2,i%2]es to generate shading

    l1 = ax[i//2,i%2].lines[0]



    # Get the xy data from the lines so that we can shade

    x1 = l1.get_xydata()[:,0]

    y1 = l1.get_xydata()[:,1]

    ax[i//2,i%2].fill_between(x1,y1, color="goldenrod", alpha=0.8)



    #grid

    ax[i//2,i%2].grid(b=True, which='major', color='grey', linewidth=0.3)

    

    ax[i//2,i%2].set_title('Distribution of {}'.format(feature), fontsize=18)

    ax[i//2,i%2].set_ylabel('count', fontsize=12)

    ax[i//2,i%2].set_xlabel('Modality', fontsize=12)



    ax[i//2,i%2].set_ylabel("frequency", fontsize=12)

    ax[i//2,i%2].set_xlabel(str(feature), fontsize=12)

    

plt.tight_layout()

plt.show()
f,ax=plt.subplots(4,2,figsize=(12,12))



for i,feature in enumerate(['sex','cp','fbs','restecg','exang','slope','ca','thal']):

    colors = ['darkturquoise']

    sns.countplot(x=feature,data=data,ax=ax[i//2,i%2], palette = colors, alpha=0.8, edgecolor=('white'), linewidth=2)

    ax[i//2,i%2].grid(b=True, which='major', color='grey', linewidth=0.2)

    ax[i//2,i%2].set_title('Count of {}'.format(feature), fontsize=18)

    ax[i//2,i%2].set_ylabel('count', fontsize=12)

    ax[i//2,i%2].set_xlabel('modality', fontsize=12)



plt.tight_layout()

plt.show()
f,ax=plt.subplots(3,2,figsize=(12,12))

f.delaxes(ax[2,1])



for i,feature in enumerate(['age','thalach','chol','trestbps','oldpeak','age']):

    sns.distplot(data[data['target']==0][(feature)], ax=ax[i//2,i%2], kde_kws={"color":"white"}, hist=False )

    sns.distplot(data[data['target']==1][(feature)], ax=ax[i//2,i%2], kde_kws={"color":"white"}, hist=False )



    # Get the two lines from the ax[i//2,i%2]es to generate shading

    l1 = ax[i//2,i%2].lines[0]

    l2 = ax[i//2,i%2].lines[1]



    # Get the xy data from the lines so that we can shade

    x1 = l1.get_xydata()[:,0]

    y1 = l1.get_xydata()[:,1]

    x2 = l2.get_xydata()[:,0]

    y2 = l2.get_xydata()[:,1]

    ax[i//2,i%2].fill_between(x2,y2, color="darkorange", alpha=0.6)

    ax[i//2,i%2].fill_between(x1,y1, color="darkturquoise", alpha=0.6)



    #grid

    ax[i//2,i%2].grid(b=True, which='major', color='grey', linewidth=0.3)

    

    ax[i//2,i%2].set_title('{} vs target'.format(feature), fontsize=18)

    ax[i//2,i%2].set_ylabel('count', fontsize=12)

    ax[i//2,i%2].set_xlabel('Modality', fontsize=12)



    #sns.despine(ax[i//2,i%2]=ax[i//2,i%2], left=True)

    ax[i//2,i%2].set_ylabel("frequency", fontsize=12)

    ax[i//2,i%2].set_xlabel(str(feature), fontsize=12)



plt.tight_layout()

plt.show()
f,ax=plt.subplots(3,2,figsize=(12,12))

f.delaxes(ax[2,1])

colors=['darkturquoise','darkorange']

for i,feature in enumerate(['age','thalach','chol','trestbps','oldpeak']):

    sns.boxplot(x='target', y=feature, data=data , ax=ax[i//2,i%2], palette=colors, boxprops=dict(alpha=0.8))

    sns.stripplot(y=feature, x='target', 

                          data=data,

                          ax=ax[i//2,i%2],

                          jitter=True, marker='o',

                          alpha=0.6, 

                          color="springgreen")



    ax[i//2,i%2].grid(b=True, which='major', color='grey', linewidth=0.2)

    ax[i//2,i%2].set_title('Count of {}'.format(feature), fontsize=18)

    ax[i//2,i%2].set_ylabel('count', fontsize=12)

    ax[i//2,i%2].set_xlabel('Modality', fontsize=12)

    

    sns.despine()

    ax[i//2,i%2].grid(b=True, which='major', color='grey', linewidth=0.4)



    ax[i//2,i%2].set_title(str(feature)+' '+'vs'+' '+'target', fontsize=18)

    ax[i//2,i%2].set_ylabel("count", fontsize=12)

    ax[i//2,i%2].set_xlabel(('target'), fontsize=12)



    plt.setp(ax[i//2,i%2].artists, edgecolor = 'white')

    plt.setp(ax[i//2,i%2].lines, color='white')



plt.tight_layout()

plt.show()
f,ax=plt.subplots(4,2,figsize=(12,12))



for i,feature in enumerate(['sex','cp','fbs','restecg','exang','slope','ca','thal']):

    colors = ['darkturquoise', 'darkorange']

    sns.countplot(x=feature,data=data,hue='target',ax=ax[i//2,i%2], palette = colors, alpha=0.7, edgecolor=('white'), linewidth=2)

    ax[i//2,i%2].grid(b=True, which='major', color='grey', linewidth=0.4)

    ax[i//2,i%2].set_title('Count of {} vs target'.format(feature), fontsize=18)

    ax[i//2,i%2].legend(loc='best')

    ax[i//2,i%2].set_ylabel('count', fontsize=12)

    ax[i//2,i%2].set_xlabel('modality', fontsize=12)



plt.tight_layout()

plt.show()
def scatterplot(var1,var2,var3,var4):

    f,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))

    #f.delaxes(ax[2,1])

    

    colors = ['darkturquoise','darkorange']

    ax1 = sns.scatterplot(x = data[var1], y = data[var2], hue = "target",

                        data = data,  ax=ax1, palette=colors, alpha=0.8, edgecolor="white",linewidth=0.1)

    ax1.grid(b=True, which='major', color='lightgrey', linewidth=0.2)

    ax1.set_title(str(var1)+' '+'vs'+' '+str(var2)+' '+'vs target', fontsize=18)

    ax1.set_xlabel(str(var1), fontsize=12)

    ax1.set_ylabel(str(var2), fontsize=12)



    ax2 = sns.scatterplot(x = data[var3], y = data[var4], hue = "target",

                        data = data,  ax=ax2, palette=colors, alpha=0.8, edgecolor="white",linewidth=0.1)

    ax2.grid(b=True, which='major', color='lightgrey', linewidth=0.2)

    ax2.set_title(str(var3)+' '+'vs'+' '+str(var4)+' '+'vs target', fontsize=18)

    ax2.set_xlabel(str(var1), fontsize=12)

    ax2.set_ylabel(str(var2), fontsize=12)
scatterplot('age','thalach','age', 'chol')

scatterplot('age','trestbps','age', 'oldpeak')

scatterplot('thalach','trestbps','thalach', 'chol')

scatterplot('thalach','oldpeak','chol', 'trestbps')

scatterplot('chol','oldpeak','trestbps', 'oldpeak')
def multivariate_count(var):

    for x in [

        #'sex',

        'cp',

        'fbs',

        #'restecg',

        #'exang',

        #'slope',

        #'ca',

        'thal']:

        ax = sns.catplot(x=x, hue="target", col=var, 

               data=data, kind="count", palette = ['darkturquoise', 'darkorange'], alpha=0.7, edgecolor=('white'), linewidth=2)

        ax.fig.suptitle(str(var)+' vs '+str(x)+' vs target', fontsize=18) 

    

        plt.subplots_adjust(top=0.8)

        plt.show()  
multivariate_count('exang')

#multivariate_count('fbs')

#multivariate_count('cp')

#multivariate_count('ca')

#multivariate_count('restecg')

#multivariate_count('slope')

#multivariate_count('thal')

multivariate_count('sex')
def multivariate_swarn(col,x,y):

    g = sns.FacetGrid(data, col=col, hue='target', palette = ['darkturquoise', 'darkorange'], height=5)

    ax = g.map(sns.swarmplot, x, y, alpha=0.7, edgecolor=('white'), linewidth=0.1)

    ax.fig.suptitle(str(x)+' vs '+str(y)+' by '+str(col)+' vs target', fontsize=18) 

    plt.subplots_adjust(top=0.8)

    plt.show()
multivariate_swarn('fbs','exang','oldpeak')

#multivariate_swarn('exang','thal','thalach')

#multivariate_swarn('sex','thal','chol')

#multivariate_swarn('slope','ca','age')



multivariate_swarn('fbs','ca','oldpeak')

#multivariate_swarn('exang','cp','thalach')

#multivariate_swarn('sex','restecg','chol')

#multivariate_swarn('slope','thal','age')



multivariate_swarn('fbs','ca','age')

#multivariate_swarn('exang','cp','oldpeak')

#multivariate_swarn('sex','restecg','thalach')

#multivariate_swarn('slope','thal','chol')



multivariate_swarn('exang','cp','age')

#multivariate_swarn('restecg','ca','oldpeak')

#multivariate_swarn('ca','sex','thalach')

#multivariate_swarn('cp','slope','chol')
data['oldpeak_x_age'] = data['oldpeak'] * data['age']

data['oldpeak_div_age'] = data['oldpeak'] / data['age']

data['ca_div_age'] = data['ca'] / data['age']

data['oldpeak_square'] = data['oldpeak'] * data['oldpeak']



data = data.fillna(0)



features = list(data)

features.remove('target')



idx = features 

for df in [data]:

    df['sum'] = df[idx].sum(axis=1)  

    df['count'] = df[idx].count(axis=1)  

    df['min'] = df[idx].min(axis=1)

    df['max'] = df[idx].max(axis=1)

    df['mean'] = df[idx].mean(axis=1)

    df['std'] = df[idx].std(axis=1)

    df['skew'] = df[idx].skew(axis=1)

    df['kurt'] = df[idx].kurtosis(axis=1)

    df['med'] = df[idx].median(axis=1)
def preprocessing(dataset, y):

    target_col = [y]

    cat_cols   = dataset.nunique()[dataset.nunique() < 5].keys().tolist()

    cat_cols   = [x for x in cat_cols ]

    #numerical columns

    num_cols   = [x for x in dataset.columns if x not in cat_cols + target_col]

    #Binary columns with 2 values

    bin_cols   = dataset.nunique()[dataset.nunique() == 2].keys().tolist()

    #Columns more than 2 values

    multi_cols = [i for i in cat_cols if i not in bin_cols]



    #Label encoding Binary columns

    le = LabelEncoder()

    for i in bin_cols :

        dataset[i] = le.fit_transform(dataset[i])



    #Duplicating columns for multi value columns

    dataset = pd.get_dummies(data = dataset,columns = multi_cols )



    #Scaling Numerical columns

    std = StandardScaler()

    scaled = std.fit_transform(dataset[num_cols])

    scaled = pd.DataFrame(scaled,columns=num_cols)



    #dropping original values merging scaled values for numerical columns

    df_dataset_og = dataset.copy()

    dataset = dataset.drop(columns = num_cols,axis = 1)

    dataset = dataset.merge(scaled,left_index=True,right_index=True,how = "left")

    return dataset
data = preprocessing(data,'target')
train_df = data

features = list(train_df)

features.remove('target')

target = train_df['target']
# Confusion matrix 

def plot_confusion_matrix(cm, classes,

                          normalize = False,

                          title = 'Confusion matrix"',

                          cmap = plt.cm.Blues) :

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)

    plt.title(title, fontsize=12)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation = 0)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :

        plt.text(j, i, cm[i, j],

                 horizontalalignment = 'center',

                 color = 'white' if cm[i, j] > thresh else 'black')

 

    plt.tight_layout()

    plt.ylabel('True label', fontsize=12)

    plt.xlabel('Predicted label', fontsize=12)
model = lgb.LGBMClassifier(**{

                'learning_rate': 0.06,

                'feature_fraction': 0.7,

                'bagging_freq': 6,

                'scale_pos_weight': 1,         

                'bagging_fraction': 0.3,

                'max_depth':-1,

                'objective': 'binary',

                'n_jobs': -1,

                'n_estimators':5000,

                'metric':'auc',

                'save_binary': True,

                'feature_fraction_seed': 42,

                'bagging_seed': 42,

                'boosting_type': 'gbdt',

                'verbose': 1,

                'is_unbalance': False,

                'boost_from_average': True

})
plt.rcParams['figure.figsize']=(6,4)



print('LGBM modeling...')

start_time = timer(None)



cms= []

tprs = []

aucs = []

y_real = []

y_proba = []

recalls = []

roc_aucs = []

f1_scores = []

accuracies = []

precisions = []



oof = np.zeros(len(train_df))

mean_fpr = np.linspace(0,1,100)

feature_importance_df = pd.DataFrame()

i = 1



folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

    print('Fold:', fold_ )

    model = model.fit(train_df.iloc[trn_idx][features], target.iloc[trn_idx],

                      eval_set = (train_df.iloc[val_idx][features], target.iloc[val_idx]),

                      verbose = 100,

                      eval_metric = 'auc',

                      early_stopping_rounds = 100)

    

    oof[val_idx] =  model.predict_proba(train_df.iloc[val_idx][features])[:,1]

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = model.feature_importances_

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    # Roc curve by fold

    f = plt.figure(1)

    fpr, tpr, t = roc_curve(train_df.target[val_idx], oof[val_idx])

    tprs.append(interp(mean_fpr, fpr, tpr))

    roc_auc = auc(fpr, tpr)

    aucs.append(roc_auc)

    plt.plot(fpr, tpr, lw=2, alpha=0.5, label='ROC fold %d (AUC = %0.4f)' % (i,roc_auc))

    

    # Precion recall by folds

    g = plt.figure(2)

    precision, recall, _ = precision_recall_curve(train_df.target[val_idx], oof[val_idx])

    y_real.append(train_df.target[val_idx])

    y_proba.append(oof[val_idx])

    plt.plot(recall, precision, lw=2, alpha=0.3, label='P|R fold %d' % (i))  

    

    i= i+1

    

    # Shap values

    explainer = shap.TreeExplainer(model)

    shap_values = shap.TreeExplainer(model).shap_values(train_df.iloc[val_idx][features])

    

    # Scores 

    roc_aucs.append(roc_auc_score(train_df.target[val_idx], oof[val_idx]))

    accuracies.append(accuracy_score(train_df.target[val_idx], oof[val_idx].round()))

    recalls.append(recall_score(train_df.target[val_idx], oof[val_idx].round()))

    precisions.append(precision_score(train_df.target[val_idx], oof[val_idx].round()))

    f1_scores.append(f1_score(train_df.target[val_idx], oof[val_idx].round()))

    

    # Confusion matrix by folds

    cms.append(confusion_matrix(train_df.target[val_idx], oof[val_idx].round()))

    

# Metrics

print(

        '\nCV roc score        : {0:.4f}, std: {1:.4f}.'.format(np.mean(roc_aucs), np.std(roc_aucs)),

        '\nCV accuracy score   : {0:.4f}, std: {1:.4f}.'.format(np.mean(accuracies), np.std(accuracies)),

        '\nCV recall score     : {0:.4f}, std: {1:.4f}.'.format(np.mean(recalls), np.std(recalls)),

        '\nCV precision score  : {0:.4f}, std: {1:.4f}.'.format(np.mean(precisions), np.std(precisions)),

        '\nCV f1 score         : {0:.4f}, std: {1:.4f}.'.format(np.mean(f1_scores), np.std(f1_scores))

)



# Timer end    

timer(start_time)



# Roc curve

f = plt.figure(1)

plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'grey')

mean_tpr = np.mean(tprs, axis=0)

mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, color='blue',

         label=r'Mean ROC (AUC = %0.4f)' % ((mean_auc)),lw=2, alpha=1)

plt.grid(b=True, which='major', color='grey', linewidth=0.4)

plt.xlabel('False Positive Rate', fontsize=12)

plt.ylabel('True Positive Rate', fontsize=12)

plt.title('LGBM - ROC by folds', fontsize=18)

plt.legend(loc="lower right")



# PR plt

g = plt.figure(2)

plt.plot([0,1],[1,0],linestyle = '--',lw = 2,color = 'grey')

y_real = np.concatenate(y_real)

y_proba = np.concatenate(y_proba)

precision, recall, _ = precision_recall_curve(y_real, y_proba)

plt.plot(recall, precision, color='blue',

         label=r'Mean P|R')

plt.grid(b=True, which='major', color='grey', linewidth=0.4)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('LGBM P|R curve by folds', fontsize=18)

plt.legend(loc="lower left")



# Confusion matrix 

plt.rcParams["axes.grid"] = False

cm = np.average(cms, axis=0)

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cm, 

                      classes=class_names, 

                      title='LGBM Confusion matrix [averaged/folds]')

plt.show()
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:37].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(10,10))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False),

            edgecolor=('white'), linewidth=2, palette="rocket")

plt.title('LGBM Features importance (averaged/folds)', fontsize=18)

plt.tight_layout()
display(

shap.force_plot(explainer.expected_value, shap_values[0,:], train_df.iloc[val_idx][features].iloc[0,:],figsize=(10, 5)),

shap.force_plot(explainer.expected_value, shap_values[1,:], train_df.iloc[val_idx][features].iloc[1,:],figsize=(10, 5)),

shap.force_plot(explainer.expected_value, shap_values[2,:], train_df.iloc[val_idx][features].iloc[2,:],figsize=(10, 5)),

shap.force_plot(explainer.expected_value, shap_values[3,:], train_df.iloc[val_idx][features].iloc[3,:],figsize=(10, 5)),

shap.force_plot(explainer.expected_value, shap_values[4,:], train_df.iloc[val_idx][features].iloc[4,:],figsize=(10, 5)),

shap.force_plot(explainer.expected_value, shap_values[5,:], train_df.iloc[val_idx][features].iloc[5,:],figsize=(10, 5)))
shap.force_plot(explainer.expected_value, shap_values, train_df.iloc[val_idx][features],figsize=(10, 5), plot_cmap='RdBu')
model = lgb.LGBMClassifier(**{

                'learning_rate': 0.06,

                'feature_fraction': 0.7,

                'bagging_freq': 6,

                'scale_pos_weight': 1.75,         

                'bagging_fraction': 0.3,

                'max_depth':-1,

                'objective': 'binary',

                'n_jobs': -1,

                'n_estimators':5000,

                'metric':'auc',

                'save_binary': True,

                'feature_fraction_seed': 42,

                'bagging_seed': 42,

                'boosting_type': 'gbdt',

                'verbose': 1,

                'is_unbalance': False,

                'boost_from_average': True

})
plt.rcParams['figure.figsize']=(6,4)



print('LGBM modeling...')

start_time = timer(None)



cms= []

tprs = []

aucs = []

y_real = []

y_proba = []

recalls = []

roc_aucs = []

f1_scores = []

accuracies = []

precisions = []



oof = np.zeros(len(train_df))

mean_fpr = np.linspace(0,1,100)

feature_importance_df = pd.DataFrame()

i = 1



folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

    print('Fold:', fold_ )

    model = model.fit(train_df.iloc[trn_idx][features], target.iloc[trn_idx],

                      eval_set = (train_df.iloc[val_idx][features], target.iloc[val_idx]),

                      verbose = 100,

                      eval_metric = 'auc',

                      early_stopping_rounds = 100)

    

    oof[val_idx] =  model.predict_proba(train_df.iloc[val_idx][features])[:,1]

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = model.feature_importances_

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    # Roc curve by fold

    f = plt.figure(1)

    fpr, tpr, t = roc_curve(train_df.target[val_idx], oof[val_idx])

    tprs.append(interp(mean_fpr, fpr, tpr))

    roc_auc = auc(fpr, tpr)

    aucs.append(roc_auc)

    plt.plot(fpr, tpr, lw=2, alpha=0.5, label='ROC fold %d (AUC = %0.4f)' % (i,roc_auc))

    

    # Precion recall by folds

    g = plt.figure(2)

    precision, recall, _ = precision_recall_curve(train_df.target[val_idx], oof[val_idx])

    y_real.append(train_df.target[val_idx])

    y_proba.append(oof[val_idx])

    plt.plot(recall, precision, lw=2, alpha=0.3, label='P|R fold %d' % (i))  

    

    i= i+1

    

    # Shap values

    explainer = shap.TreeExplainer(model)

    shap_values = shap.TreeExplainer(model).shap_values(train_df.iloc[val_idx][features])

    

    # Scores 

    roc_aucs.append(roc_auc_score(train_df.target[val_idx], oof[val_idx]))

    accuracies.append(accuracy_score(train_df.target[val_idx], oof[val_idx].round()))

    recalls.append(recall_score(train_df.target[val_idx], oof[val_idx].round()))

    precisions.append(precision_score(train_df.target[val_idx], oof[val_idx].round()))

    f1_scores.append(f1_score(train_df.target[val_idx], oof[val_idx].round()))

    

    # Confusion matrix by folds

    cms.append(confusion_matrix(train_df.target[val_idx], oof[val_idx].round()))

    

# Metrics

print(

        '\nCV roc score        : {0:.4f}, std: {1:.4f}.'.format(np.mean(roc_aucs), np.std(roc_aucs)),

        '\nCV accuracy score   : {0:.4f}, std: {1:.4f}.'.format(np.mean(accuracies), np.std(accuracies)),

        '\nCV recall score     : {0:.4f}, std: {1:.4f}.'.format(np.mean(recalls), np.std(recalls)),

        '\nCV precision score  : {0:.4f}, std: {1:.4f}.'.format(np.mean(precisions), np.std(precisions)),

        '\nCV f1 score         : {0:.4f}, std: {1:.4f}.'.format(np.mean(f1_scores), np.std(f1_scores))

)



# Timer end    

timer(start_time)



# Roc curve

f = plt.figure(1)

plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'grey')

mean_tpr = np.mean(tprs, axis=0)

mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, color='blue',

         label=r'Mean ROC (AUC = %0.4f)' % ((mean_auc)),lw=2, alpha=1)

plt.grid(b=True, which='major', color='grey', linewidth=0.4)

plt.xlabel('False Positive Rate', fontsize=12)

plt.ylabel('True Positive Rate', fontsize=12)

plt.title('LGBM - ROC by folds', fontsize=18)

plt.legend(loc="lower right")



# PR plt

g = plt.figure(2)

plt.plot([0,1],[1,0],linestyle = '--',lw = 2,color = 'grey')

y_real = np.concatenate(y_real)

y_proba = np.concatenate(y_proba)

precision, recall, _ = precision_recall_curve(y_real, y_proba)

plt.plot(recall, precision, color='blue',

         label=r'Mean P|R')

plt.grid(b=True, which='major', color='grey', linewidth=0.4)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('LGBM P|R curve by folds', fontsize=18)

plt.legend(loc="lower left")



# Confusion matrix 

plt.rcParams["axes.grid"] = False

cm = np.average(cms, axis=0)

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cm, 

                      classes=class_names, 

                      title='LGBM Confusion matrix [averaged/folds]')

plt.show()
model = cb.CatBoostClassifier(**{

                                'learning_rate':0.05,

                                'max_depth':2,

                                'n_estimators':100,

                                'eval_metric': 'AUC',

                                'bootstrap_type': 'Bayesian',

                                'use_best_model':True,

                                'bagging_temperature': 1,

                                'objective': 'Logloss',

                                'od_type': 'Iter',

                                'l2_leaf_reg': 2,

                                'allow_writing_files': False})
plt.rcParams['figure.figsize']=(6,4)



print('CatBoost modeling...')

start_time = timer(None)



cms= []

tprs = []

aucs = []

y_real = []

y_proba = []

recalls = []

roc_aucs = []

f1_scores = []

accuracies = []

precisions = []



oof = np.zeros(len(train_df))

mean_fpr = np.linspace(0,1,100)

feature_importance_df = pd.DataFrame()

i = 1



folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

    print('Fold:', fold_ )

    model = model.fit(train_df.iloc[trn_idx][features], target.iloc[trn_idx],

                      eval_set = (train_df.iloc[val_idx][features], target.iloc[val_idx]),

                      verbose = 100,

                      early_stopping_rounds=100)

    

    oof[val_idx] =  model.predict_proba(train_df.iloc[val_idx][features])[:,1]

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = model.feature_importances_

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    # Roc curve by fold

    f = plt.figure(1)

    fpr, tpr, t = roc_curve(train_df.target[val_idx], oof[val_idx])

    tprs.append(interp(mean_fpr, fpr, tpr))

    roc_auc = auc(fpr, tpr)

    aucs.append(roc_auc)

    plt.plot(fpr, tpr, lw=2, alpha=0.5, label='ROC fold %d (AUC = %0.4f)' % (i,roc_auc))

    

    # Precion recall by folds

    g = plt.figure(2)

    precision, recall, _ = precision_recall_curve(train_df.target[val_idx], oof[val_idx])

    y_real.append(train_df.target[val_idx])

    y_proba.append(oof[val_idx])

    plt.plot(recall, precision, lw=2, alpha=0.3, label='P|R fold %d' % (i))  

    

    i= i+1

    

    # Scores 

    roc_aucs.append(roc_auc_score(train_df.target[val_idx], oof[val_idx]))

    accuracies.append(accuracy_score(train_df.target[val_idx], oof[val_idx].round()))

    recalls.append(recall_score(train_df.target[val_idx], oof[val_idx].round()))

    precisions.append(precision_score(train_df.target[val_idx], oof[val_idx].round()))

    f1_scores.append(f1_score(train_df.target[val_idx], oof[val_idx].round()))

    

    # Confusion matrix by folds

    cms.append(confusion_matrix(train_df.target[val_idx], oof[val_idx].round()))

    

# Metrics

print(

        '\nCV roc score        : {0:.4f}, std: {1:.4f}.'.format(np.mean(roc_aucs), np.std(roc_aucs)),

        '\nCV accuracy score   : {0:.4f}, std: {1:.4f}.'.format(np.mean(accuracies), np.std(accuracies)),

        '\nCV recall score     : {0:.4f}, std: {1:.4f}.'.format(np.mean(recalls), np.std(recalls)),

        '\nCV precision score  : {0:.4f}, std: {1:.4f}.'.format(np.mean(precisions), np.std(precisions)),

        '\nCV f1 score         : {0:.4f}, std: {1:.4f}.'.format(np.mean(f1_scores), np.std(f1_scores))

)



# Timer end    

timer(start_time)



# Roc curve

f = plt.figure(1)

plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'grey')

mean_tpr = np.mean(tprs, axis=0)

mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, color='blue',

         label=r'Mean ROC (AUC = %0.4f)' % ((mean_auc)),lw=2, alpha=1)

plt.grid(b=True, which='major', color='grey', linewidth=0.4)

plt.xlabel('False Positive Rate', fontsize=12)

plt.ylabel('True Positive Rate', fontsize=12)

plt.title('LGBM - ROC by folds', fontsize=18)

plt.legend(loc="lower right")



# PR plt

g = plt.figure(2)

plt.plot([0,1],[1,0],linestyle = '--',lw = 2,color = 'grey')

y_real = np.concatenate(y_real)

y_proba = np.concatenate(y_proba)

precision, recall, _ = precision_recall_curve(y_real, y_proba)

plt.plot(recall, precision, color='blue',

         label=r'Mean P|R')

plt.grid(b=True, which='major', color='grey', linewidth=0.4)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('CB P|R curve by folds', fontsize=18)

plt.legend(loc="lower left")



# Confusion matrix 

plt.rcParams["axes.grid"] = False

cm = np.average(cms, axis=0)

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cm, 

                      classes=class_names, 

                      title='CB Confusion matrix [averaged/folds]')

plt.show()
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:37].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(10,10))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False),

            edgecolor=('white'), linewidth=2, palette="rocket")

plt.title('CB - Features importance (averaged/folds)', fontsize=18)

plt.tight_layout()