import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



print(os.listdir("../input/cleanedloanfico/"))
def univariate(df,col,vartype,hue =None):

    

    '''

    Univariate function will plot the graphs based on the parameters.

    df      : dataframe name

    col     : Column name

    vartype : variable type : continuos or categorical

                Continuos(0)   : Distribution, Violin & Boxplot will be plotted.

                Categorical(1) : Countplot will be plotted.

    hue     : It's only applicable for categorical analysis.

    

    '''

    sns.set(style="darkgrid")

    

    if vartype == 0:

        fig, ax=plt.subplots(nrows =1,ncols=3,figsize=(20,8))

        ax[0].set_title("Distribution Plot")

        sns.distplot(df[col],ax=ax[0])

        ax[1].set_title("Violin Plot")

        sns.violinplot(data =df, x=col,ax=ax[1], inner="quartile")

        ax[2].set_title("Box Plot")

        sns.boxplot(data =df, x=col,ax=ax[2],orient='v')

    

    if vartype == 1:

        temp = pd.Series(data = hue)

        fig, ax = plt.subplots()

        width = len(df[col].unique()) + 6 + 4*len(temp.unique())

        fig.set_size_inches(width , 7)

        ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue) 

        if len(temp.unique()) > 0:

            for p in ax.patches:

                ax.annotate('{:1.1f}%'.format((p.get_height()*100)/float(len(loan))), (p.get_x()+0.05, p.get_height()+20))  

        else:

            for p in ax.patches:

                ax.annotate(p.get_height(), (p.get_x()+0.32, p.get_height()+20)) 

        del temp

    else:

        exit

        

    plt.show()
def crosstab(df,col):

    '''

    df : Dataframe

    col: Column Name

    '''

    crosstab = pd.crosstab(df[col], df['loan_status'],margins=True)

    crosstab['Probability_Charged Off'] = round((crosstab['Charged Off']/crosstab['All']),3)

    crosstab = crosstab[0:-1]

    max1 = max(crosstab['Probability_Charged Off'])

    maxx = crosstab.loc[crosstab['Probability_Charged Off']==max1]

    

    return crosstab,maxx
# Probability of charge off

def bivariate_prob(df,col,stacked= True):

    '''

    df      : Dataframe

    col     : Column Name

    stacked : True(default) for Stacked Bar

    '''

    # get dataframe from crosstab function

    plotCrosstab,maxx = crosstab(df,col)

    

    linePlot = plotCrosstab[['Probability_Charged Off']]      

    barPlot =  plotCrosstab.iloc[:,0:2]

    ax = linePlot.plot(figsize=(20,8), marker='o',color = 'b')

    ax2 = barPlot.plot(kind='bar',ax = ax,rot=1,secondary_y=True,stacked=stacked)

    ax.set_title(df[col].name.title()+' vs Probability Charge Off',fontsize=20,weight="bold")

    ax.set_xlabel(df[col].name.title(),fontsize=14)

    ax.set_ylabel('Probability of Charged off',color = 'b',fontsize=14)

    ax2.set_ylabel('Number of Applicants',color = 'g',fontsize=14)

    plt.show()
loan = pd.read_csv('../input/cleanedloanfico/cleanedData.csv')
# Drop

drop = ['last_fico_range_high','last_fico_range_low']

loan.drop(drop, axis=1, inplace=True)

r,c=loan.shape

print(f"The number of rows {r}\nThe number of columns {c}")

loan.dropna(axis=0, how = 'any', inplace = True)

r1,c1=loan.shape

print(f"The difference between earlier and dropped Nan rows: {r-r1}")
loan['emp_length'] = loan['emp_length'].replace({'1 year':1,'10+ years':'10','2 years':2,'3 years':3,"4 years":4,"5 years":5,

                                                 "6 years":6,"7 years":7,"8 years":8,"9 years":9,"< 1 year":0})



#a_dataframe.drop(a_dataframe[a_dataframe.B > 3].index, inplace=True)



loan.drop(loan[loan['emp_length']=="Self-Employed"].index,inplace = True)

loan['emp_length'] = loan['emp_length'].apply(pd.to_numeric)
univariate(df=loan,col='emp_length',vartype=1)
# average fico score from range

#loan["fico_score"] = (loan['fico_range_high'] + loan['fico_range_low'])/2

loan['fico_score'] = loan[['fico_range_low', 'fico_range_high']].mean(axis=1)

drop = ['fico_range_low','fico_range_high']

loan.drop(drop, axis=1, inplace=True)

loan['fico_score']
loan.head(5)
fico,maxx = crosstab(loan,'fico_score')

display(fico)



print("maximum")

display(maxx)



bivariate_prob(df=loan,col='fico_score')
mask = (loan.loan_status == 'Charged Off')

loan['target'] = 0

loan.loc[mask,'target'] = 1
del loan['loan_status']
loan.loc[loan['target']==0]
loan.loc[loan['target']==1]
loan['target'].value_counts()
loan.dtypes
categorical = loan.columns[loan.dtypes == 'object']

categorical
X = pd.get_dummies(loan[loan.columns], columns=categorical).astype(float)

y = loan['target']
X
if 'target' in X:

    del X['target']

X.columns
y
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from imblearn.over_sampling import SMOTE

import pickle

scaler = preprocessing.MinMaxScaler()

#X_scaled = scaler.fit_transform(X)

#X_scaledd = pd.DataFrame(X_scaled, columns=X.columns)



#X_scaled = preprocessing.scale(X)

#print(X_scaled)

#print('   ')

#print(X_scaled.shape)

#X_scaledd

#X_scaledd.isnull().sum()
def run_models(X_train, y_train, X_test, y_test, model_type = 'Non-balanced'):

    

    clfs = {'GradientBoosting': GradientBoostingClassifier(verbose=1,max_depth= 8, n_estimators=150, max_features = 0.3),

            'LogisticRegression' : LogisticRegression(verbose=1,C=10.0,solver='saga',penalty = 'elasticnet',l1_ratio = 1),

            #'GaussianNB': GaussianNB(),

            'RandomForestClassifier': RandomForestClassifier(verbose=1,n_estimators=10,criterion='entropy') #10

            }

    cols = ['model','accuracy_score','matthews_corrcoef', 'roc_auc_score', 'precision_score', 'recall_score','f1_score']



    models_report = pd.DataFrame(columns = cols)

    conf_matrix = dict()



    for clf, clf_name in zip(clfs.values(), clfs.keys()):

        

        cross_val_score(clf, X_train, y_train , cv=5)

        

        clf.fit(X_train, y_train)

        

        filename = f'{clf_name}_{model_type}.sav'

        pickle.dump(clf, open(filename, 'wb'))

        

        y_pred = clf.predict(X_test)

        y_score = clf.predict_proba(X_test)[:,1]



        print('computing {} - {} '.format(clf_name, model_type))



        tmp = pd.Series({'model_type': model_type,

                         'model': clf_name,

                         'accuracy_score': metrics.accuracy_score(y_test, y_pred),

                         'roc_auc_score' : metrics.roc_auc_score(y_test, y_score),

                         'matthews_corrcoef': metrics.matthews_corrcoef(y_test, y_pred),

                         'precision_score': metrics.precision_score(y_test, y_pred),

                         'recall_score': metrics.recall_score(y_test, y_pred),

                         'f1_score': metrics.f1_score(y_test, y_pred)})



        models_report = models_report.append(tmp, ignore_index = True)

        conf_matrix[clf_name] = pd.crosstab(y_test, y_pred, rownames=['True'], colnames= ['Predicted'], margins=False)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, drop_intermediate = False, pos_label = 1)



        plt.figure(1, figsize=(6,6))

        plt.xlabel('false positive rate')

        plt.ylabel('true positive rate')

        plt.title('ROC curve - {}'.format(model_type))

        plt.plot(fpr, tpr, label = clf_name )

        plt.legend(loc=2, prop={'size':11})

    plt.plot([0,1],[0,1], color = 'black')

    

    return models_report, conf_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.4, random_state=0, shuffle = True)

X_train_sc = scaler.fit_transform(X_train)

X_test_sc = scaler.fit_transform(X_test)



#scores = cross_val_score(clf, X_scaled, y , cv=5, scoring='roc_auc')



%time models_report, conf_matrix = run_models(X_train_sc, y_train, X_test_sc, y_test, model_type = 'Non-balanced')
X_Train_scaled = pd.DataFrame(X_train_sc, columns=X.columns)

X_Train_scaled
X_Test_scaled = pd.DataFrame(X_test_sc, columns=X.columns)

X_Test_scaled
X_Train_scaled.describe()
X_Test_scaled.describe()
models_report
conf_matrix['LogisticRegression']
conf_matrix['GradientBoosting']
conf_matrix['RandomForestClassifier']
p = X.iloc[20]

loan.iloc[20]
k=p.array

dd = dict(zip(X.columns,k))

p = pd.DataFrame(dd, columns = p.index, index=[0])
p
import pickle

loaded_model = pickle.load(open('LogisticRegression_Non-balanced.sav', 'rb'))

result = loaded_model.predict(p)

prob = loaded_model.predict_proba(p)

print("Target",result[0],"\nProbability",prob)
loaded_model = pickle.load(open('RandomForestClassifier_Non-balanced.sav', 'rb'))

result = loaded_model.predict(p) 

prob = loaded_model.predict_proba(p)

print("Target",result[0],"\nProbability",prob)
loaded_model = pickle.load(open('GradientBoosting_Non-balanced.sav', 'rb'))

result = loaded_model.predict(p)

prob = loaded_model.predict_proba(p)

print("Target",result[0],"\nProbability",prob)