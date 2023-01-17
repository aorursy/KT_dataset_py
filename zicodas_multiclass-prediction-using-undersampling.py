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
df  = pd.read_csv('/kaggle/input/fetal-health-classification/fetal_health.csv')

df.head()
df.columns
df.info()
df.describe(include='all').T
df.isnull().sum()
for i in df.columns:

    #print('no of unique vlaues in {} column {}'.format(i,df[i].nunique()))

    if df[i].nunique() <=3:

        print(i)
df['fetal_health'].value_counts()
import seaborn as sns

sns.countplot('fetal_health',data=df)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, StratifiedKFold



print('Normal',round(df['fetal_health'].value_counts()[1]/len(df) * 100), '% of the data ')

print('Suspect',round(df['fetal_health'].value_counts()[2]/len(df) * 100), '% of the data ')

print('Pathological',round(df['fetal_health'].value_counts()[3]/len(df) * 100), '% of the data ')



X = df.drop('fetal_health',axis=1)

y = df['fetal_health']





sss = StratifiedKFold(n_splits=6,random_state=1,shuffle=True)



for train_index,test_index in sss.split(X,y):

    #print('Train:', train_index)

    #print('Test:',test_index)

    

    original_Xtrain,original_Xtest = X.iloc[train_index],X.iloc[test_index]

    original_ytrain,original_ytest = y.iloc[train_index],y.iloc[test_index]



# Turn into an array

original_Xtrain = original_Xtrain.values

original_Xtest = original_Xtest.values

original_ytrain = original_ytrain.values

original_ytest = original_ytest.values



# See if both train and test label distribution are similarly distributed

train_unique_label,train_counts_label = np.unique(original_ytrain,return_counts=True)

test_unique_label,test_counts_label = np.unique(original_ytest,return_counts=True)



print(train_unique_label,train_counts_label)

print(test_unique_label,test_counts_label)



print('Label Distributions:')

print(train_counts_label/len(original_ytrain))

print(test_counts_label/len(original_ytest))
df = df.sample(frac=1)

#amount of pathological classes 176 rows



Path_df = df.loc[df['fetal_health']==3.0]

susp_df = df.loc[df['fetal_health']==2.0][:176]

norm_df = df.loc[df['fetal_health']==1.0][:176]



normal_distributed_df = pd.concat([Path_df,susp_df,norm_df])

#print(normal_distributed_df.head(5))



#Shuffle dataframe rows

new_df = normal_distributed_df.sample(frac=1, random_state=42)



print(new_df.head(5))
new_df.shape
import matplotlib.pyplot as plt

print('Distribution of classes in the sample distribution')

print(new_df['fetal_health'].value_counts()/len(new_df))



sns.countplot('fetal_health',data=new_df)

plt.title('Equally Distirbuted Classes',fontsize=14)

plt.show()
#Correlation Matrix



f, (ax1,ax2) = plt.subplots(2,1,figsize=(24,20))



#Entire DataFrame

corr = df.corr()

sns.heatmap(corr,cmap='coolwarm_r',annot_kws={'size':20},ax=ax1)

ax1.set_title("Imbalanced Correlation Matrix \n (dont use Reference)", fontsize=14)



sub_sample_corr = new_df.corr()

sns.heatmap(sub_sample_corr,cmap='coolwarm_r',annot_kws={'size':20},ax=ax2)

ax2.set_title("SubSample Correlation Matrix \n (use Reference)",fontsize=14)



plt.show()
#Highest correlated pairs

c = new_df.corr().abs()

s = c.unstack()

so = s.sort_values(kind="quicksort")

so
#BoxPlots

#Negative correlated values are compared with the class(label)(The lower our feature value the more likely it will be apthological):



f, axes = plt.subplots(ncols=4,figsize=(20,4))



sns.boxplot(x='fetal_health',y='accelerations',data=new_df,ax=axes[0])

axes[0].set_title('accelerations vs fetal_health Negative Corr')



sns.boxplot(x='fetal_health',y='histogram_mean',data=new_df,ax=axes[1])

axes[1].set_title('histogram_mean vs fetal_health Negative Corr')



sns.boxplot(x='fetal_health',y='histogram_median',data=new_df,ax=axes[2])

axes[2].set_title('histogram_median vs fetal_health Negative Corr')



sns.boxplot(x='fetal_health',y='histogram_mode',data=new_df,ax=axes[3])

axes[3].set_title('histogram_mode vs fetal_health Negative Corr')



plt.show()
# Positive correlations (The higher the feature the probability increases that it will be a )

sns.boxplot(x="fetal_health", y="abnormal_short_term_variability", data=new_df)

plt.title('abnormal_short_term_variability vs fetal_health Positive Correlation')

plt.show()
histogram_mean_fhealth = new_df['histogram_mean'].loc[new_df['fetal_health']==1].values

q25,q75 = np.percentile(histogram_mean_fhealth,25),np.percentile(histogram_mean_fhealth,75)



print('Quartile 25: {} | Quartile 75: {}'.format(q25,q75))

histogram_mean_iqr = q75 - q25

print('iqr: {}'.format(histogram_mean_iqr))



histogram_mean_cut_off = histogram_mean_iqr * 1.5

histogram_mean_lower,histogram_mean_upper = q25 - histogram_mean_cut_off, q75 + histogram_mean_cut_off



print('cutoff: {}'.format(histogram_mean_cut_off))

print('histogram_mean Lower: {}'.format(histogram_mean_lower))

print('histogram_mean_upper: {}'.format(histogram_mean_upper))



outliers = [x for x in histogram_mean_fhealth if x < histogram_mean_lower or x > histogram_mean_upper]

print('Features histogram_mean outliers for fhealth cases: {}'.format(len(outliers)))

print('histogram_mean outliers:{}'.format(outliers))



new_df = new_df.drop(new_df[(new_df['histogram_mean'] > histogram_mean_upper) | (new_df['histogram_mean'] < histogram_mean_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_df)))

print('----' * 44)
histogram_median_fhealth = new_df['histogram_median'].loc[new_df['fetal_health']==1].values

q25,q75 = np.percentile(histogram_median_fhealth,25),np.percentile(histogram_median_fhealth,75)



print('Quartile 25: {} | Quartile 75: {}'.format(q25,q75))

histogram_median_iqr = q75 - q25

print('iqr: {}'.format(histogram_median_iqr))



histogram_median_cut_off = histogram_median_iqr * 1.5

histogram_median_lower,histogram_median_upper = q25 - histogram_median_cut_off, q75 + histogram_median_cut_off



print('cutoff: {}'.format(histogram_median_cut_off))

print('histogram_median Lower: {}'.format(histogram_median_lower))

print('histogram_median_upper: {}'.format(histogram_median_upper))



outliers = [x for x in histogram_median_fhealth if x < histogram_median_lower or x > histogram_median_upper]

print('Features histogram_median outliers for fhealth cases: {}'.format(len(outliers)))

print('histogram_median outliers:{}'.format(outliers))



new_df = new_df.drop(new_df[(new_df['histogram_median'] > histogram_median_upper) | (new_df['histogram_median'] < histogram_median_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_df)))

print('----' * 44)
## -----> histogram_mode Removing Outliers



histogram_mode_fhealth = new_df['histogram_mode'].loc[new_df['fetal_health']==1].values

q25,q75 = np.percentile(histogram_mode_fhealth,25),np.percentile(histogram_mode_fhealth,75)



print('Quartile 25: {} | Quartile 75: {}'.format(q25,q75))

histogram_mode_iqr = q75 - q25

print('iqr: {}'.format(histogram_mode_iqr))



histogram_mode_cut_off = histogram_mode_iqr * 1.5

histogram_mode_lower,histogram_mode_upper = q25 - histogram_mode_cut_off, q75 + histogram_mode_cut_off



print('cutoff: {}'.format(histogram_mode_cut_off))

print('histogram_mode Lower: {}'.format(histogram_mode_lower))

print('histogram_mode_upper: {}'.format(histogram_mode_upper))



outliers = [x for x in histogram_mode_fhealth if x < histogram_mode_lower or x > histogram_mode_upper]

print('Features histogram_mode outliers for fhealth cases: {}'.format(len(outliers)))

print('histogram_mode outliers:{}'.format(outliers))



new_df = new_df.drop(new_df[(new_df['histogram_mode'] > histogram_mode_upper) | (new_df['histogram_mode'] < histogram_mode_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_df)))

print('----' * 44)
## -----> abnormal_short_term_variability Removing Outliers



abnormal_short_term_variability_fhealth = new_df['abnormal_short_term_variability'].loc[new_df['fetal_health']==1].values

q25,q75 = np.percentile(abnormal_short_term_variability_fhealth,25),np.percentile(abnormal_short_term_variability_fhealth,75)



print('Quartile 25: {} | Quartile 75: {}'.format(q25,q75))

abnormal_short_term_variability_iqr = q75 - q25

print('iqr: {}'.format(abnormal_short_term_variability_iqr))



abnormal_short_term_variability_cut_off = abnormal_short_term_variability_iqr * 1.5

abnormal_short_term_variability_lower,abnormal_short_term_variability_upper = q25 - abnormal_short_term_variability_cut_off, q75 + abnormal_short_term_variability_cut_off



print('cutoff: {}'.format(abnormal_short_term_variability_cut_off))

print('abnormal_short_term_variability Lower: {}'.format(abnormal_short_term_variability_lower))

print('abnormal_short_term_variability_upper: {}'.format(abnormal_short_term_variability_upper))



outliers = [x for x in abnormal_short_term_variability_fhealth if x < abnormal_short_term_variability_lower or x > abnormal_short_term_variability_upper]

print('Features abnormal_short_term_variability outliers for fhealth cases: {}'.format(len(outliers)))

print('abnormal_short_term_variability outliers:{}'.format(outliers))



new_df = new_df.drop(new_df[(new_df['abnormal_short_term_variability'] > abnormal_short_term_variability_upper) | (new_df['abnormal_short_term_variability'] < abnormal_short_term_variability_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_df)))

print('----' * 44)
# no outliers present

f,(ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,6))



colors = ['#B3F9C5', '#f9c5b3']

# Boxplots with outliers removed

# Feature histogram_mean

sns.boxplot(x="fetal_health", y="histogram_mean", data=new_df,ax=ax1, palette=colors)

ax1.set_title("histogram_mean Feature \n Reduction of outliers", fontsize=14)





# Feature histogram_median

sns.boxplot(x="fetal_health", y="histogram_median", data=new_df, ax=ax2, palette=colors)

ax2.set_title("histogram_median Feature \n Reduction of outliers", fontsize=14)



# Feature histogram_mode

sns.boxplot(x="fetal_health", y="histogram_mode", data=new_df, ax=ax3, palette=colors)

ax3.set_title("histogram_mode Feature \n Reduction of outliers", fontsize=14)





# Feature abnormal_short_term_variability

sns.boxplot(x="fetal_health", y="abnormal_short_term_variability", data=new_df, ax=ax4, palette=colors)

ax4.set_title("abnormal_short_term_variability Feature \n Reduction of outliers", fontsize=14)





plt.show()
#undersampling before cross validating

X = new_df.drop('fetal_health',axis=1)

y = new_df['fetal_health']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
# Classifier Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

#from sklearn import LDA

#from sklearn import QDA

import collections



classifiers = {"Logistic Regression":LogisticRegression(),

               "KNearest":KNeighborsClassifier(),

               "DecisionTreeClassifier":DecisionTreeClassifier(max_depth=5),

               "RandomForestClassifier":RandomForestClassifier(max_depth=5),

               "AdaBoostClassifier":AdaBoostClassifier(),

               "GaussianNB":GaussianNB()}
from sklearn.model_selection import cross_val_score

import warnings

warnings.filterwarnings("ignore")



for key,classifier in classifiers.items():

    classifier.fit(X_train,y_train)

    training_score = cross_val_score(classifier,X_train,y_train,cv=5)

    print("Models:",classifier.__class__.__name__,"has a training score of",round(training_score.mean(),2)*100,"accuracy score")

    
log_reg_score = cross_val_score(LogisticRegression(),X_train,y_train,cv=5)

print("Logistic Regression cross validation score",round(log_reg_score.mean()*100,2).astype(str)+'%')



knears_score = cross_val_score(KNeighborsClassifier(),X_train,y_train,cv=5)

print("Knearest neighbors cross vlaidation score",round(knears_score.mean()*100,2).astype(str)+'%')



Rand_score = cross_val_score(RandomForestClassifier(max_depth=5),X_train,y_train,cv=5)

print("Random Forest cross validation score",round(Rand_score.mean()*100,2).astype(str)+'%')



tree_score = cross_val_score(DecisionTreeClassifier(max_depth=5),X_train,y_train,cv=5)

print("Decisiontree Classifiers",round(tree_score.mean()*100,2).astype(str)+'%')



Ada_score = cross_val_score(AdaBoostClassifier(),X_train,y_train,cv=5)

print("AdaBoost Classifiers",round(Ada_score.mean()*100,2).astype(str)+'%')



Gauss_score = cross_val_score(GaussianNB(),X_train,y_train,cv=5)

print("GaussianNB Classifiers",round(Gauss_score.mean()*100,2).astype(str)+'%')
from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

from collections import Counter



#We will undersample during cross validating

undersample_X = df.drop('fetal_health',axis=1)

undersample_y = df['fetal_health']



for train_index,test_index in sss.split(undersample_X,undersample_y):

    undersample_Xtrain,undersample_Xtest = undersample_X.iloc[train_index],undersample_X.iloc[test_index]

    undersample_ytrain,undersample_ytest = undersample_y.iloc[train_index],undersample_y.iloc[test_index]



    #print(sss.get_n_splits(undersample_X,undersample_y))

    

    

undersample_Xtrain = undersample_Xtrain.values

undersample_ytrain = undersample_ytrain.values

undersample_Xtest = undersample_Xtest.values

undersample_ytest = undersample_ytest.values



undersample_accuracy = []

undersample_precision = []

undersample_recall = []

undersample_f1 = []

undersample_auc = []



# implementing the near miss technique

# Demonstration of distribution by nearmiss

X_nearmiss,y_nearmiss = NearMiss().fit_sample(undersample_X.values,undersample_y.values)

print("nearmiss label distribution: {}".format(Counter(y_nearmiss)))



from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report,roc_curve

for train, test in sss.split(undersample_Xtrain, undersample_ytrain):

    undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='auto'), DecisionTreeClassifier(max_depth=5)) # SMOTE happens during Cross Validation not before..

    undersample_model = undersample_pipeline.fit(undersample_Xtrain[train], undersample_ytrain[train])

    undersample_prediction = undersample_model.predict(undersample_Xtrain[test])

    undersample_prediction_prob = undersample_model.predict_proba(undersample_Xtrain[test])

    

    undersample_accuracy.append(undersample_pipeline.score(original_Xtrain[test], original_ytrain[test]))

    undersample_precision.append(precision_score(original_ytrain[test], undersample_prediction,average='weighted'))

    undersample_recall.append(recall_score(original_ytrain[test], undersample_prediction,average='weighted'))

    undersample_f1.append(f1_score(original_ytrain[test], undersample_prediction,average='weighted'))

    undersample_auc.append(roc_auc_score(original_ytrain[test], undersample_prediction_prob,multi_class='ovr'))

    

    fprate, tprate, thresholds = roc_curve(original_ytrain[test], undersample_prediction,pos_label=1)
print(undersample_accuracy)

print(undersample_precision)

print(undersample_recall)

print(undersample_f1)

print(undersample_auc)
''' Learning curve helps to visualize the training score and the cross validation score

If cross vlaidation score is trending more towards a high training score then a larger dataset is required for cross validation of algorithm'''



# Let's Plot LogisticRegression Learning Curve

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import learning_curve



def plot_learning_curve(estimator1, estimator2, estimator3, estimator4, estimator5, estimator6, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    f, ((ax1, ax2), (ax3, ax4),(ax5,ax6)) = plt.subplots(3,2, figsize=(20,14), sharey=True)

    plt.subplots_adjust(wspace=0.4)

    if ylim is not None:

        plt.ylim(*ylim)



    # First Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax1.set_title("Logistic Regression Learning Curve", fontsize=10)

    ax1.set_xlabel('Training size (m)')

    ax1.set_ylabel('Score')

    ax1.grid(True)

    ax1.legend(loc="best")

    

    # Second Estimator 

    train_sizes, train_scores, test_scores = learning_curve(

        estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax2.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax2.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax2.set_title("Knears Neighbors Learning Curve", fontsize=10)

    ax2.set_xlabel('Training size (m)')

    ax2.set_ylabel('Score')

    ax2.grid(True)

    ax2.legend(loc="best")

    

    # Third Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax3.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax3.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax3.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax3.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax3.set_title("Random Forest Classifier Learning Curve", fontsize=10)

    ax3.set_xlabel('Training size (m)')

    ax3.set_ylabel('Score')

    ax3.grid(True)

    ax3.legend(loc="best")

    

    # Fourth Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax4.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax4.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax4.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax4.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax4.set_title("Decision Tree Classifier Learning Curve", fontsize=10)

    ax4.set_xlabel('Training size (m)')

    ax4.set_ylabel('Score')

    ax4.grid(True)

    ax4.legend(loc="best")

    

    # Fifth Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator5, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax5.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax5.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax5.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax5.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax5.set_title("Ada Boost Classifier Learning Curve", fontsize=10)

    ax5.set_xlabel('Training size (m)')

    ax5.set_ylabel('Score')

    ax5.grid(True)

    ax5.legend(loc="best")

    

    # Sixth Estimator

    train_sizes, train_scores, test_scores = learning_curve(

        estimator6, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    ax6.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="#ff9124")

    ax6.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")

    ax6.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",

             label="Training score")

    ax6.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",

             label="Cross-validation score")

    ax6.set_title("Gaussian NB Classifier Learning Curve", fontsize=10)

    ax6.set_xlabel('Training size (m)')

    ax6.set_ylabel('Score')

    ax6.grid(True)

    ax6.legend(loc="best")

    

    

    

    return plt
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

plot_learning_curve(LogisticRegression(),KNeighborsClassifier(),RandomForestClassifier(max_depth=5),DecisionTreeClassifier(max_depth=5),AdaBoostClassifier(),GaussianNB(), X_train, y_train, (0.60, 1.01), cv=cv, n_jobs=4)
'''Example of Receiver Operating Characteristic (ROC) metric to evaluate classifier output quality.



ROC curves typically feature true positive rate on the Y axis, and false positive rate on the X axis. This means that the top left corner of the plot is the “ideal” point - a false positive rate of zero, and a true positive rate of one. This is not very realistic, but it does mean that a larger area under the curve (AUC) is usually better.



The “steepness” of ROC curves is also important, since it is ideal to maximize the true positive rate while minimizing the false positive rate.

'''



from sklearn.metrics import roc_curve

from sklearn.model_selection import cross_val_predict



log_reg_pred = cross_val_predict(LogisticRegression(),X_train,y_train,cv=5,method='decision_function')

knears_pred = cross_val_predict(KNeighborsClassifier(),X_train,y_train,cv=5,method='predict_proba')

Rfc_pred = cross_val_predict(RandomForestClassifier(max_depth=5),X_train,y_train,cv=5,method='predict_proba')

Dec_tree_pred = cross_val_predict(DecisionTreeClassifier(max_depth=5),X_train,y_train,cv=5,method='predict_proba')

ada_pred = cross_val_predict(AdaBoostClassifier(),X_train,y_train,cv=5,method='predict_proba')

gauss_pred = cross_val_predict(GaussianNB(),X_train,y_train,cv=5,method='predict_proba')



print((log_reg_pred))

print((knears_pred))

print((Rfc_pred))

print((Dec_tree_pred))

print((ada_pred))

print((gauss_pred))

from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score

Random = RandomForestClassifier(max_depth=5)

Random.fit(X_train,y_train)

y_pred = Random.predict(X_train)



#overfitting case:



print('----'*45)

print('Overfitting: \n')

print('Recall Score: {:.2f}'.format(recall_score(y_train,y_pred,average='weighted')))

print('Precision Score: {:.2f}'.format(precision_score(y_train,y_pred,average='weighted')))

print('f1 score: {:.2f}'.format(f1_score(y_train,y_pred,average='weighted')))

print('accuracy score: {:.2f}'.format(accuracy_score(y_train,y_pred)))



#Real work

print('----'*45)



print("Accuracy score: {:.2f}".format(np.mean(undersample_accuracy)))

print("Precision score: {:.2f}".format(np.mean(undersample_precision)))

print("F1 score: {:.2f}".format(np.mean(undersample_f1)))

print("Recall score: {:.2f}".format(np.mean(undersample_recall)))

print('---' * 45)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_train, y_pred)
sns.heatmap(confusion_matrix(y_train, y_pred), annot=True,cmap='Blues', fmt='g');