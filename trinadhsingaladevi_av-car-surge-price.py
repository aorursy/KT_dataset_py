import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train_df = pd.read_csv('train.csv')

test_df = pd.read_csv('test.csv')
print(train_df.shape)

train_df.head()
train_df.isnull().sum()
print("Surge Pricing type: ",train_df['Surge_Pricing_Type'].unique())

print("Type_of_Cab: ",train_df['Type_of_Cab'].unique())

print("Confidence_Life_Style_Index: ",train_df['Confidence_Life_Style_Index'].unique())
train_df.describe()
#FUNCTION FOR PROVIDING FEATURE SUMMARY

def feature_summary(df_fa):

    print('DataFrame shape')

    print('rows:',df_fa.shape[0])

    print('cols:',df_fa.shape[1])

    col_list=['Null','Unique_Count','Data_type','Max/Min','Mean','Std','Skewness','Sample_values']

    df=pd.DataFrame(index=df_fa.columns,columns=col_list)

    df['Null']=list([len(df_fa[col][df_fa[col].isnull()]) for i,col in enumerate(df_fa.columns)])

    #df['%_Null']=list([len(df_fa[col][df_fa[col].isnull()])/df_fa.shape[0]*100 for i,col in enumerate(df_fa.columns)])

    df['Unique_Count']=list([len(df_fa[col].unique()) for i,col in enumerate(df_fa.columns)])

    df['Data_type']=list([df_fa[col].dtype for i,col in enumerate(df_fa.columns)])

    for i,col in enumerate(df_fa.columns):

        if 'float' in str(df_fa[col].dtype) or 'int' in str(df_fa[col].dtype):

            df.at[col,'Max/Min']=str(round(df_fa[col].max(),2))+'/'+str(round(df_fa[col].min(),2))

            df.at[col,'Mean']=df_fa[col].mean()

            df.at[col,'Std']=df_fa[col].std()

            df.at[col,'Skewness']=df_fa[col].skew()

        df.at[col,'Sample_values']=list(df_fa[col].unique())

           

    return(df.fillna('-'))
#train=pd.read_csv(path1+'application_train.csv')

print('application_train Feature Summary')

with pd.option_context('display.max_rows',train_df.shape[1]):

    train_fs=feature_summary(train_df) 
train_fs
nominal_var = list(train_fs[train_fs['Data_type'] == 'object'].index)

print("Total categorical columns are:",len(nominal_var))

print(nominal_var)
ord_var = list(train_fs[train_fs['Data_type'] == 'int64'].index)

print("Total ordinal columns are:",len(ord_var))

print(ord_var)
cont_var = train_fs[train_fs['Data_type'] == 'float64'].index

print("Total categorical columns are:",len(cont_var))

print(cont_var)
f,ax = plt.subplots(1,2,figsize=(18,8))

train_df['Surge_Pricing_Type'].value_counts().plot.pie(autopct='%1.1f%%',ax = ax[0])

graph = sns.countplot('Surge_Pricing_Type',data = train_df,ax = ax[1])

i=1

for p in graph.patches:

    print(p)

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,train_df['Surge_Pricing_Type'].value_counts()[i],ha="center")

    i += 1
target = 'Surge_Pricing_Type'

nominal_var.remove('Trip_ID')

ord_var.remove('Surge_Pricing_Type')


f,ax = plt.subplots(len(nominal_var),2,figsize = (18,18))

#data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

for i in range(len(nominal_var)):

    train_df[[nominal_var[i],target]].groupby([nominal_var[i]]).mean().plot.bar(ax = ax[i][0])

    sns.countplot(nominal_var[i],data = train_df,ax = ax[i][1],hue=target)
f,ax = plt.subplots(len(ord_var),2,figsize = (18,18))

#data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

for i in range(len(ord_var)):

    train_df[ord_var[i]].value_counts().plot.bar(ax = ax[i][0])

    #data['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

    sns.countplot(ord_var[i],data = train_df,ax = ax[i][1],hue=target)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
train_fs
df1=train_df



#df1.mode()
def drop_col(data):

    data = data.drop(['Trip_ID','Var1','Life_Style_Index'],axis = 1)

    return data
def fill_null(data):

    data['Type_of_Cab'] = data['Type_of_Cab'].fillna('B')

    data['Customer_Since_Months'] = data['Customer_Since_Months'].fillna('10')

    data['Confidence_Life_Style_Index'] = data['Confidence_Life_Style_Index'].fillna('B')

    #print(data)

    return data
def lable_encod(le,data,nominal_var):

    for col in nominal_var:

        data[col]=le.fit_transform(data[col])

    return data
#find the mode to fill the null values

df1 = drop_col(df1)

#print(df1.mode())

df1 = fill_null(df1)

df1 = lable_encod(le,df1,nominal_var)
sns.heatmap(df1.corr(),annot = True,cmap='RdYlGn',linewidths=0.2)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
from sklearn.metrics import confusion_matrix 

from sklearn.model_selection import train_test_split 

from sklearn.svm import SVC 

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn import preprocessing
X = df1.drop([target],axis = 1) 

y = df1[target]
def Scaling(X):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X)

    return X_train_scaled 
X_train_scaled = Scaling(X)
# dividing X, y into train and test data 

X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y, random_state = 0,test_size = 0.3)
X_train.shape
#SVM

svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 

svm_predictions = svm_model_linear.predict(X_test) 

  

# model accuracy for X_test   

accuracy = svm_model_linear.score(X_test, y_test) 

  

# creating a confusion matrix 

cm = confusion_matrix(y_test, svm_predictions)


print(accuracy)

sns.heatmap(cm,annot=True,fmt = '')
from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 

  

# accuracy on X_test 

accuracy = knn.score(X_test, y_test) 

print(accuracy)

  

# creating a confusion matrix 

knn_predictions = knn.predict(X_test)  

cm = confusion_matrix(y_test, knn_predictions)
sns.heatmap(cm,annot=True,fmt = '')
params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
# Performing CV to tune parameters for best SVM fit 

svm_model = GridSearchCV(SVC(), params_grid, cv=5)

svm_model.fit(X_train, y_train)
# View the accuracy score

print('Best score for training data:', svm_model.best_score_,"\n") 



# View the best parameters for the model found using grid search

print('Best C:',svm_model.best_estimator_.C,"\n") 

print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")

print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")



final_model = svm_model.best_estimator_

Y_pred = final_model.predict(X_test_scaled)

Y_pred_label = list(encoder.inverse_transform(Y_pred))
print(test_df.shape)

test_df.head()
test_df.isnull().sum()
test_df1 = test_df

#find the mode to fill the null values

test_df1 = drop_col(test_df1)

#print(df1.mode())

test_df1 = fill_null(test_df1)

test_df1 = lable_encod(le,test_df1,nominal_var)
X_test_scaled = Scaling(test_df1)
sub = pd.DataFrame(test_df[['Trip_ID']])
#SVM

svm_values = svm_model_linear.predict(X_test_scaled)

sub['Surge_Pricing_Type'] = svm_values

sub.to_csv('svmlinear.csv',index = False)
#KNN

knn_values = knn.predict(X_test_scaled)

sub['Surge_Pricing_Type'] = knn_values

sub.to_csv('knnlinear.csv',index = False)