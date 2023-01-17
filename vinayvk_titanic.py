# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import pandas as pd
import numpy as np
import seaborn as sns
import cufflinks as cf
import matplotlib.pyplot as plt
%matplotlib inline
print(os.listdir("../input/"))
df=pd.read_csv("../input/train.csv")
df.head()



sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.countplot(x='Survived',data=df,hue='Sex',palette='RdBu_r')
sns.countplot(x='Survived',data=df,hue='Pclass',palette='RdBu_r')
sns.countplot(x='Survived',data=df,hue='Pclass')

df['Age'].plot.hist(bins=35)
sns.distplot(df['Age'].dropna(),kde=False,bins=30)
df.info()
sns.countplot(x='SibSp',data=df)
df['Fare'].hist(bins=40,figsize=(10,4))

cf.go_online()
df['Fare'].plot()
base_dataset=df
def nullvalue_function(base_dataset,percentage):
    
    # Checking the null value occurance
    
    print(base_dataset.isna().sum())

    # Printing the shape of the data 
    
    print(base_dataset.shape)
    
    # Converting  into percentage table
    
    null_value_table=pd.DataFrame((base_dataset.isna().sum()/base_dataset.shape[0])*100).sort_values(0,ascending=False )
    
    null_value_table.columns=['null percentage']
    
    # Defining the threashold values 
    
    null_value_table[null_value_table['null percentage']>percentage].index
    
    # Drop the columns that has null values more than threashold 
    base_dataset.drop(null_value_table[null_value_table['null percentage']>percentage].index,axis=1,inplace=True)
    
    # Replace the null values with median() # continous variables 
    for i in base_dataset.describe().columns:
        base_dataset[i].fillna(base_dataset[i].median(),inplace=True)
    # Replace the null values with mode() #categorical variables
    for i in base_dataset.describe(include='object').columns:
        base_dataset[i].fillna(base_dataset[i].value_counts().index[0],inplace=True)
  
    print(base_dataset.shape)
    
    return base_dataset
def impute_average(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 35
        elif Pclass == 2:
            return 28
        else:
            return 28
    else:
        return Age
df.head()
df['Age'] = df[['Age','Pclass']].apply(impute_average,axis=1)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age',data=df)
nulltreateddataset=nullvalue_function(df,30)
nulltreateddataset.dropna(inplace = True)

nulltreateddataset.head()
sns.heatmap(nulltreateddataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')
from sklearn import preprocessing

def variables_creation(base_dataset,unique):
    import numpy as np
    
    cat=base_dataset.describe(include='object').columns
    
    cont=base_dataset.describe().columns
    
    x=[]
    
    for i in base_dataset[cat].columns:
        if len(base_dataset[i].value_counts().index)<unique:
            x.append(i)
    
    dummies_table=pd.get_dummies(base_dataset[x],drop_first=True)#added drop_first clause
    encode_table=base_dataset[x]
    
    le = preprocessing.LabelEncoder()
    lable_encode=[]
    
    for i in encode_table.columns:
        le.fit(encode_table[i])
        le.classes_
        lable_encode.append(le.transform(encode_table[i]))
        
    lable_encode=np.array(lable_encode)
    lable=lable_encode.reshape(base_dataset.shape[0],len(x))
    lable=pd.DataFrame(lable)
    return (lable,dummies_table,cat,cont)
lable,dummies,cat,cont = variables_creation(nulltreateddataset,8)
cat
Pclass=pd.get_dummies(nulltreateddataset['Pclass'],drop_first=True)
ad=pd.concat([dummies,nulltreateddataset,Pclass],axis=1)
ad.columns
ad.head()
ad.sort_values(by='Fare').head(2)
cols=ad.columns.tolist()
cols.sort
cols = [2,3,'Sex_male','Embarked_Q','Embarked_S','Age','SibSp','Parch', 'Fare','Survived',]
ad=ad[cols]
ad.head(3)

sns.boxplot(ad['Fare'])
def outliers(df):
    import numpy as np
    import statistics as sts

    for i in df.describe().columns:
        x=np.array(df[i])
        p=[]
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1
        LTV= Q1 - (1.5 * IQR)
        UTV= Q3 + (1.5 * IQR)
        for j in x:
            if j <= LTV or j>=UTV:
                p.append(sts.median(x))
            else:
                p.append(j)
        df[i]=p
    return df
ad=outliers(ad)
sns.boxplot(ad['Fare'])
#Preparing Target and Response sets of data
x = ad.drop('Survived',axis=1)
y = ad['Survived']
x_cols = x.columns
df1 = pd.DataFrame(x, columns=x_cols)
df1.head(2)
#Splitting the dataset into Test and Train
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(df1, y,train_size=0.8, random_state=42)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred_class=logreg.predict(x_test)
len(y_pred_class)
x_test.shape
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred_class))
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))
#Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_class))
logreg.predict(x_test)[0:10]
logreg.predict_proba(x_test)[0:10, :]
y_pred_prob = logreg.predict_proba(x_test)[:, 1]
Probab_Matrix=pd.DataFrame(logreg.predict_proba(x_test))
Probab_Matrix.head()
import matplotlib.pyplot as plt 
plt.hist(logreg.predict_proba(x_test))
df=confusion_matrix(y_test, y_pred_class)
#Function to detect all interpretations from Confusion Matrix
def confusion_matrix(df):
    TN=df[1][1]
    FP=df[0][1]
    TP=df[0][0]
    FN=df[0][1]
    print(TP,FP,FN,TN)
    print("*****************Set 1:  Acutals as Base (Vertical) ***************************************")
    
    print("recall/True Positive Rate/Sensitivity(TPR)",(TP)/(TP+FN))
    TPR=(TP)/(TP+FN)
    FPR = (FP)/(TN+FP)
    FNR = (FN)/(TP+FN)
    TNR = (TN)/(TN+FP)
    PPV = (TP)/(TP+FP)
    LRP = TPR/FPR
    LRN = FNR/TNR
    print("False Negative Rate(FNR)",(FN)/(TP+FN))
    print("Specificity/True Negative Rate(TNR)",(TN)/(TN+FP))
    print("False Positive Rate(FPR)",(FP)/(TN+FP))
    
    
    print("*********************Set 2 : Predicted as Base (Horizontal)***************************************")
    
    print("Positive Predicted Value/Precision(PPV)",(TP)/(TP+FP))
    print("False Discovery Rate(FDR)",(FP)/(TP+FP))
    print("False Omission Rate(FOR)",(FN)/(TN+FN))
    print("Negative Predicted Value(NPV)",(TN)/(TN+FN))

        
    print("************************************************************")
    print("Lkelihood Ratio Positive/LRP",TPR/FPR)
    print("Lkelihood Ratio Negative/LRN",FNR/TNR)
    print("Diagnostics Odds Ratio",LRP/LRN)
    print("F1",(2/((1/TPR)+(1/PPV))))
confusion_matrix(df)
#ROC:The Reciever Operating Characteristic (ROC) curve graphs the true positive rate versus the false positive rate:

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Predicting Survival of Passengers')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

#AUC is the percentage of the ROC plot that is underneath the curve:
# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(y_test, y_pred_prob))
# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])

# calculate cross-validated AUC

from sklearn.model_selection import cross_val_score
cross_val_score(logreg, x, y, cv=10, scoring='roc_auc').mean()
evaluate_threshold(0.5)