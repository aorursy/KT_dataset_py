import math
import csv
import numpy as np
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import seaborn as sns
    
application_data = pd.read_csv("../input/application_train.csv")
corr = application_data.corr().abs()

def FeatureAnalysis(df,corr):
    
    corr_target = corr['TARGET']
    for colName in df:
        
        if(colName=='TARGET' or colName=='SK_ID_CURR'):
            continue
        if (colName  in corr_target.index):
            pass
        else:
            continue
            
        print("Feature Name: ",colName)
        print("Coefficient Correlation: ",corr_target[colName])
        print("Correlation Rank: ",corr['TARGET'].rank(ascending=False).loc[colName],"/",corr['TARGET'].count())
        print("DataType: ",df[colName].dtypes)
        if(df[colName].dtypes!='object' and len(df[colName].unique())<=20 ):
            print("Feature Type: ","Categorical")
            if(len(df[colName].unique()) == 2):
                print("Recommendation: ","Label Encoding")
            else:
                print("Recommendation: ","One Hot Encoding")
            dfTemp = pd.DataFrame({'Target_0' :df[application_data['TARGET']==0].groupby([colName])[colName].count()}).reset_index()
            dfTemp2 = pd.DataFrame({'Target_1' :df[application_data['TARGET']==1].groupby([colName])[colName].count()}).reset_index()
            dfTemp = dfTemp.merge(dfTemp2,on=colName,how='outer')
            dfTemp.plot(kind='bar',xticks=dfTemp.index,x=colName,figsize=(12, 4),rot=1)
            plt.xticks(np.arange(len(dfTemp.index)), dfTemp[colName], rotation='vertical')
            plt.show()

        else:
            print("Feature Type: ","Numerical")
            print("Recommendation: ","Normalize")

            if(df[colName].min()<0):
                print("Has Negative Values !")
        print("Unique Values: ",len(df[colName].unique()))
        print("Null Values: ",df[colName].isnull().values.sum())

        sns.kdeplot(application_data.loc[application_data['TARGET'] == 0, colName], label = 'target == 0')
        sns.kdeplot(application_data.loc[application_data['TARGET'] == 1, colName], label = 'target == 1')
        plt.show()
        df[colName].hist()
        plt.show()

FeatureAnalysis(application_data,corr)
