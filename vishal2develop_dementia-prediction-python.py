import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# For Imputation
from sklearn.preprocessing import LabelEncoder

# For data preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_validate

# For model building
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# For visualizing the descision tree
from sklearn import tree





from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#import the necessary modelling algos.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score,auc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             log_loss,
                             roc_auc_score,
                             roc_curve,
                             confusion_matrix)
from sklearn.model_selection import (cross_val_score,
                                     GridSearchCV,
                                     RandomizedSearchCV,
                                     learning_curve,
                                     validation_curve,
                                     train_test_split)

from sklearn.pipeline import make_pipeline # For performing a series of operations

from sklearn.metrics import plot_confusion_matrix
df = pd.read_csv('../input/mri-and-alzheimers/oasis_longitudinal.csv')
df.head()
df.shape
# getting a feel of the data types of the columns

df.info()
df.isnull().sum()
df.describe() # for numerical cols
df.skew()
df.MMSE.fillna(df.MMSE.median(),inplace=True)
df.SES.fillna(df.SES.median(),inplace=True)
df.isnull().sum()
df.drop(columns='Hand',axis=1,inplace=True)
df.head()
df.isnull().sum()
# Reversing using mapping
ses_map = {5:1,4:2,3:3,2:4,1:5}
df.SES = df.SES.map(ses_map)

df.head()
df.SES.value_counts()
df_copy = df.copy()
df.to_csv('oasis_longitude.csv')
df.dtypes
gender_map = {'M':0, 'F':1}
df['Gender'] = df['M/F'].map(gender_map)
df.tail()
df.dtypes
df.drop(columns='M/F',axis=1,inplace=True)
df.head()
df.Group.value_counts()
target_map = {'Nondemented':0,'Demented':1,'Converted':2}

df['Group'] = df.Group.map(target_map)
df.Group.value_counts()
corr = df.corr()
corr
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the figure
fig, ax = plt.subplots(figsize=(12,8))

# Generate a custom colormap
cmap = sns.diverging_palette(250, 10, s=80, l=55, n=9, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio (mask to not display upper triangle part)
sns.heatmap(corr, mask=mask, cmap=cmap, ax=ax, annot=True);
plt.savefig('corr.png')
plt.figure(figsize=(12,8))
sns.countplot(df['CDR'])
plt.title('Distribution of CDR Levels')
plt.xlabel('CDR LEVEL')
plt.ylabel('COUNT')
plt.savefig('CDR_distribution.png')
sns.factorplot(x='CDR',y='SES',data=df,kind='box',size=5,aspect=1)
a = df.SES.value_counts()
list(a.index)

# Create list of indicies of SES counts
ses_count = df['SES'].value_counts()
ses_indexes = list(ses_count.index)

# Plot of distribution of scores for building categories
plt.figure(figsize=(12, 8))

# Plot each building
for s in ses_indexes:
    # Select the SES category
    subset = df[df['SES'] == s]
    
    # Density plot of CDR scores
    sns.kdeplot(subset['CDR'],
               label = s, shade = False, alpha = 0.8);
    
# label the plot
plt.xlabel('CDR Score', size = 20);
plt.ylabel('Density', size = 20); 
plt.title('Density Plot of CDR Scores by SES', size = 28);
plt.savefig('SES_CDR.png')
sns.factorplot(x='CDR',kind='count',col='SES',data=df)
df.EDUC.value_counts()
df.dtypes
# Create list of indicies of SES counts
edu_count= df['EDUC'].value_counts()
edu_index = list(edu_count.index)

# Plot of distribution of scores for building categories
plt.figure(figsize=(12, 8))

# Plot each building
for el in edu_index:
    # Select the SES category
    subset = df[df['EDUC'] == el]
    
    # Density plot of CDR scores
    sns.kdeplot(subset['CDR'],
               label = el, shade = False, alpha = 0.8,bw=0.5);
    
# label the plot
plt.xlabel('CDR Score', size = 20); plt.ylabel('Density', size = 20); 
plt.title('Density Plot of CDR Scores by Years of Education', size = 28);
#plt.xlim([0,2]);
plt.savefig('EDU_CDR.png')
# Min and Max years of education among subjects
min_edu = df.loc[df['EDUC']==12]
max_edu = df.loc[df['EDUC']==16]

# Stack them into a combine dataframe
edu_concat = pd.concat([min_edu,max_edu])
edu_concat.head()
# Create list of indicies of SES counts
edu_= edu_concat['EDUC'].value_counts()
edu_index = list(edu_.index)

# Plot of distribution of scores for building categories
plt.figure(figsize=(12, 8))

# Plot each building
for el in edu_index:
    # Select the SES category
    subset = edu_concat[edu_concat['EDUC'] == el]
    
    # Density plot of CDR scores
    sns.kdeplot(subset['CDR'],
               label = el, shade = False, alpha = 0.8);
    
# label the plot
plt.xlabel('CDR Score', size = 20); plt.ylabel('Density', size = 20); 
plt.title('Density Plot of CDR Scores by Years of Education', size = 28);
#plt.xlim([0,2]);
plt.savefig('EDU_CDR.png')
# Create list of indicies of SES counts
gender_count= df['Gender'].value_counts()
gender_indicies = list(gender_count.index)

# Plot of distribution of scores for building categories
plt.figure(figsize=(12, 10))

# Plot each building
for g in gender_indicies:
    # Select the SES category
    subset = df[df['Gender']==g]
    
    # Density plot of CDR scores
    sns.kdeplot(subset['CDR'],
               label = g, shade = False, alpha = 0.8);
    
# label the plot
plt.xlabel('CDR Score', size = 20); plt.ylabel('Density', size = 20); 
plt.title('Density Plot of CDR Scores by Gender', size = 28, );
plt.savefig('Gender_CDR.png')
fig = plt.figure(figsize=(12,8))
sns.catplot(x='CDR',y='Age',data=df,hue='Gender')
plt.savefig('Age_CDR.png')
fig = plt.figure(figsize=(12,8))
sns.catplot(x='CDR',y='MMSE',data=df, hue='Gender')
plt.savefig('MMSE_CDR')
fig = plt.figure(figsize=(12,8))
sns.catplot(x='CDR',y='eTIV',data=df)
fig = plt.figure(figsize=(12,8))
sns.catplot(x='CDR',y='ASF',data=df)
fig = plt.figure(figsize=(12,8))
sns.catplot(x='CDR', y='nWBV', data=df)
df.shape
df.head()
selected_df = df.drop(['Subject ID','MRI ID','CDR'],axis=1)

selected_df.head()
# Rename columns
rename_cols_dict = {'EDUC':'Education',
                   'Group':'Diagnosis'}
selected_df.rename(rename_cols_dict,axis=1,inplace=True)
selected_df.head()
selected_df.dtypes
target = selected_df.Diagnosis.values

predictors_df = selected_df.drop(['Diagnosis'],axis=1)
predictors_df.head()
predictors_df.dtypes
plt.figure(figsize=(12,8))
sns.countplot(selected_df.Diagnosis)
plt.title('Distribution of Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('COUNT')
plt.savefig('Diagnosis_distribution.png')
x_train,x_test,y_train,y_test = train_test_split(predictors_df,target,test_size=0.2,stratify=target,random_state=1)
print("Training Data - Predictors",x_train.shape)
print("Testing Data - Predictors",x_test.shape)
print("Training Data - Target",y_train.shape)
print("Testing Data - Target",y_test.shape)
from sklearn.pipeline import make_pipeline # For performing a series of operations

from sklearn.metrics import plot_confusion_matrix

from sklearn.preprocessing import StandardScaler
# Build random forest classifier
methods_data = {'Original': (x_train,y_train)}

for method in methods_data.keys():
    pip_rf = make_pipeline(StandardScaler(),
                           RandomForestClassifier(n_estimators=500,
                                                  class_weight="balanced",
                                                  random_state=123))
    hyperparam_grid = {
        "randomforestclassifier__n_estimators": [10, 50, 100, 500],
        "randomforestclassifier__max_features": ["sqrt", "log2", 0.4, 0.5],
        "randomforestclassifier__min_samples_leaf": [1, 3, 5],
        "randomforestclassifier__criterion": ["gini", "entropy"]}
    
    gs_rf = GridSearchCV(pip_rf,
                         hyperparam_grid,
                         scoring="f1_macro",
                         cv=10,
                         n_jobs=-1)
    
    gs_rf.fit(methods_data[method][0], methods_data[method][1])
    
    print("\033[1m" + "\033[0m" + "The best hyperparameters for {} data:".format(method))
    for hyperparam in gs_rf.best_params_.keys():
        print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_rf.best_params_[hyperparam])
    
    print("\033[1m" + "\033[94m" + "Best 10-folds CV f1-score: {:.2f}%.".format((gs_rf.best_score_) * 100))
# Refit RF classifier using best params
clf_rf = make_pipeline(StandardScaler(),
                       RandomForestClassifier(n_estimators=10,
                                              criterion="gini",
                                              max_features=0.4,
                                              min_samples_leaf=3,
                                              class_weight="balanced",
                                              n_jobs=-1,
                                              random_state=123))


clf_rf.fit(x_train, y_train)
# Build Gradient Boosting classifier
pip_gb = make_pipeline(StandardScaler(),
                       GradientBoostingClassifier(loss="deviance",
                                                  random_state=123))

hyperparam_grid = {"gradientboostingclassifier__max_features": ["log2", 0.5],
                   "gradientboostingclassifier__n_estimators": [100, 300, 500],
                   "gradientboostingclassifier__learning_rate": [0.001, 0.01, 0.1],
                   "gradientboostingclassifier__max_depth": [1, 2, 3]}

gs_gb = GridSearchCV(pip_gb,
                      param_grid=hyperparam_grid,
                      scoring="f1_macro",
                      cv=10,
                      n_jobs=-1)

gs_gb.fit(x_train, y_train)

print("\033[1m" + "\033[0m" + "The best hyperparameters:")
print("-" * 25)
for hyperparam in gs_gb.best_params_.keys():
    print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_gb.best_params_[hyperparam])

print("\033[1m" + "\033[94m" + "Best 10-folds CV f1-score: {:.2f}%.".format((gs_gb.best_score_) * 100))
# Build logistic model classifier
pip_logmod = make_pipeline(StandardScaler(),
                           LogisticRegression(class_weight="balanced"))

hyperparam_range = np.arange(0.5, 20.1, 0.5)

hyperparam_grid = {"logisticregression__penalty": ["l1", "l2"],
                   "logisticregression__C":  hyperparam_range,
                   "logisticregression__fit_intercept": [True, False]
                  }

gs_logmodel = GridSearchCV(pip_logmod,
                           hyperparam_grid,
                           scoring="accuracy",
                           cv=2,
                           n_jobs=-1)

gs_logmodel.fit(x_train, y_train)

print("\033[1m" + "\033[0m" + "The best hyperparameters:")
print("-" * 25)
for hyperparam in gs_logmodel.best_params_.keys():
    print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_logmodel.best_params_[hyperparam])

print("\033[1m" + "\033[94m" + "Best 10-folds CV f1-score: {:.2f}%.".format((gs_logmodel.best_score_) * 100))
estimators = {"RF": clf_rf,
              "LR": gs_logmodel,
              "GBT": gs_gb
             }

# Print out accuracy score on test data
print("The accuracy rate on test data are:")
for estimator in estimators.keys():
    print("{}: {:.2f}%".format(estimator,
        accuracy_score(y_test, estimators[estimator].predict(x_test)) * 100
          ))
predictions = gs_gb.predict(x_test)
predictions.shape
selected_df.Diagnosis.value_counts()
model_names=['RandomForestClassifier','Logistic Regression','GradientBoostingClassifier']
models = [clf_rf,gs_logmodel,gs_gb]
def compare_models(model):
    clf=model
    clf.fit(x_train,y_train)
    pred=clf.predict(x_test)
    
    # Calculating various metrics
    
    acc.append(accuracy_score(pred,y_test))
    #prec.append(precision_score(pred,y_test))
    #rec.append(recall_score(pred,y_test))
    #auroc.append(roc_auc_score(pred,y_test))
acc=[]
prec=[]
rec=[]
auroc=[]
for model in models:
    compare_models(model)

d={'Modelling Algo':model_names,'Accuracy':acc}
met_df=pd.DataFrame(d)
met_df
