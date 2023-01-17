import warnings

warnings.simplefilter('ignore')
# We import all the necessary libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
!pip install feature-engine
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,ExtraTreesRegressor

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler

from feature_engine.categorical_encoders import RareLabelCategoricalEncoder,OneHotCategoricalEncoder
train=pd.read_csv('/kaggle/input/big-mart-sales-dataset/Train_UWu5bXk.csv')

test=pd.read_csv('/kaggle/input/big-mart-sales-dataset/Test_u94Q5KV.csv')
train.head()
train.describe()
test.head()
test.describe()
train.info()
train.isnull().mean()
test.isnull().mean()
cat_variables=train.select_dtypes(include='object')



cat_variables.drop(columns='Item_Identifier',inplace=True)
def boxplot(x,y,**kwargs):

    sns.boxplot(x=x,y=y)
# We plot the box plot for all the categorical columns except for Item Identifier as it has high sparsity it needs cleaning before we can plot

f=pd.melt(train,id_vars='Item_Outlet_Sales',value_vars=cat_variables)



g=sns.FacetGrid(f,col='variable',sharey=True,col_wrap=3,height=2,sharex=False,size=5)



g=g.map(boxplot,'value','Item_Outlet_Sales')
sns.pairplot(train)
plt.figure(figsize=(6,6))

sns.barplot(x='Outlet_Size',y='Item_Outlet_Sales',hue='Outlet_Type',data=train)
plt.figure(figsize=(6,6))

sns.barplot(x='Outlet_Establishment_Year',y='Item_Outlet_Sales',hue='Outlet_Type',data=train)
plt.figure(figsize=(6,6))

sns.barplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',hue='Outlet_Type',data=train)
train.head()
# We split the item identifier so as to classify the product much easily

train['Item_Classification']=train['Item_Identifier'].str[:2]
# We perform the same in the test set as well

test['Item_Classification']=test['Item_Identifier'].str[:2]
# We create the x and y values for the train set

X=train



X=X.drop(columns='Item_Outlet_Sales')
y=train['Item_Outlet_Sales']
# We create the train and test set

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
pd.crosstab(index=X_train['Outlet_Location_Type'],columns=X_train['Outlet_Type'],values=X_train['Outlet_Size'],aggfunc=pd.Series.mode)
# We fill the outlet size with the information we got from the above table, where grocery store and supermarket type1 have size as small 

X_train.loc[X_train['Outlet_Type'] == 'Grocery Store','Outlet_Size'] = X_train.loc[X_train['Outlet_Type'] == 'Grocery Store', 'Outlet_Size'].fillna('Small')



X_train.loc[X_train['Outlet_Type'] == 'Supermarket Type1','Outlet_Size'] = X_train.loc[X_train['Outlet_Type'] == 'Supermarket Type1', 'Outlet_Size'].fillna('Small')
# We fill the X_test values by the same method

X_test.loc[X_test['Outlet_Type'] == 'Grocery Store','Outlet_Size'] = X_test.loc[X_test['Outlet_Type'] == 'Grocery Store', 'Outlet_Size'].fillna('Small')



X_test.loc[X_test['Outlet_Type'] == 'Supermarket Type1','Outlet_Size'] = X_test.loc[X_test['Outlet_Type'] == 'Supermarket Type1', 'Outlet_Size'].fillna('Small')
# We perform the same operation on the test set



test.loc[test['Outlet_Type'] == 'Grocery Store','Outlet_Size'] = test.loc[test['Outlet_Type'] == 'Grocery Store', 'Outlet_Size'].fillna('Small')



test.loc[test['Outlet_Type'] == 'Supermarket Type1','Outlet_Size'] = test.loc[test['Outlet_Type'] == 'Supermarket Type1', 'Outlet_Size'].fillna('Small')
# We groupby the weight column with respect to item type and item fat content by which we can weight of each product 

X_train['Item_Weight']=X_train.groupby(['Item_Fat_Content','Item_Type'])['Item_Weight'].apply(lambda x :x.fillna(x.mean()))



X_test['Item_Weight']=X_test.groupby(['Item_Fat_Content','Item_Type'])['Item_Weight'].apply(lambda x :x.fillna(x.mean()))
X_test['Item_Weight']=X_test['Item_Weight'].fillna(test['Item_Weight'].mean())
# We performt the same for the test dataset 



test['Item_Weight']=test.groupby(['Item_Fat_Content','Item_Type'])['Item_Weight'].apply(lambda x :x.fillna(x.mean()))
test.isnull().mean()
# we perform the mean imputation to fill the final missing value

test['Item_Weight']=test['Item_Weight'].fillna(test['Item_Weight'].mean())
# We the values 0 in item visibilty with 25th quantile value with the assumption that they have lowest visibilty

X_train.loc[X_train['Item_Visibility']==0,'Item_Visibility']=np.quantile(train['Item_Visibility'],0.25)



X_test.loc[X_test['Item_Visibility']==0,'Item_Visibility']=np.quantile(train['Item_Visibility'],0.25)
# We perform the same in the test dataset



test.loc[test['Item_Visibility']==0,'Item_Visibility']=np.quantile(train['Item_Visibility'],0.25)
# We create a new type of item fat content with respect to item classification which is non-consumable

X_train.loc[X_train['Item_Classification']=='NC','Item_Fat_Content']='Non Consumable'



X_test.loc[X_test['Item_Classification']=='NC','Item_Fat_Content']='Non Consumable'
# We perform the same procedure the on the test set

test.loc[test['Item_Classification']=='NC','Item_Fat_Content']='Non Consumable'
X_train['Item_Fat_Content'].value_counts()
# We can see from the column that there are mulitple type with similar meaning, which we replace

X_train['Item_Fat_Content']=X_train['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})



X_test['Item_Fat_Content']=X_test['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})
# We perform the same in the test set



test['Item_Fat_Content']=test['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})
# We create a list of categorical varibles to understand the sparsity

cat_variables=list(X_train.select_dtypes(include='object'))



cat_variables.remove('Item_Identifier')
cat_variables
# We plot the graph to find the values with low percentage

count=1

plt.figure(figsize=(20,10))

for col in cat_variables:

    

    temp=pd.Series(X_train[col].value_counts()/len(X_train))

    

    # make plot with the above percentages

    plt.subplot(3,3,count)

    fig = temp.sort_values(ascending=False).plot.bar()

    fig.set_xlabel(col)



    # add a line at 5 % to flag the threshold for rare categories

    fig.axhline(y=0.05, color='red')

    fig.set_ylabel('Percentage Count')

    count+=1
# We enable rare label for the outlet type

rc=RareLabelCategoricalEncoder(tol=0.05,n_categories=10,variables=['Item_Type'])



rc.fit(X_train)



X_train=rc.transform(X_train)



X_test=rc.transform(X_test)
# We perform the same in the test set 



test=rc.transform(test)
# We change the astype of establishment year to string

X_train['Outlet_Establishment_Year']=X_train['Outlet_Establishment_Year'].astype('str')



X_test['Outlet_Establishment_Year']=X_test['Outlet_Establishment_Year'].astype('str')
# We perform the same in the test set



test['Outlet_Establishment_Year']=test['Outlet_Establishment_Year'].astype('str')
# We drop the item identifier column from the dataset

X_train=X_train.drop(columns='Item_Identifier')



X_test=X_test.drop(columns='Item_Identifier')
# We perform the same the in the test dataset

test=test.drop(columns='Item_Identifier')
np.random.seed(seed=0)
# We perform one hot categorical encoding

ohce=OneHotCategoricalEncoder(drop_last=True)



ohce.fit(X_train)



X_train=ohce.transform(X_train)



X_test=ohce.transform(X_test)
# We do the same with the test set



test=ohce.transform(test)
# We scale the values in the dataset



col=list(X_train.columns)



sc=StandardScaler()



sc.fit(X_train)



X_train=pd.DataFrame(sc.transform(X_train),columns=col)



X_test=pd.DataFrame(sc.transform(X_test),columns=col)
# We perform the same in the dataset



test=pd.DataFrame(sc.transform(test),columns=col)
# We perform the linear regression on the model



regressor_lc=LinearRegression()



regressor_lc.fit(X_train,y_train)



y_pred_lc=regressor_lc.predict(X_test)
# We check the accuracy of the model



mse=mean_squared_error(y_pred_lc,y_test)



r2=r2_score(y_pred_lc,y_test)



print('The Mean Squared error is {}\nThe r2 Score is {}'.format(np.sqrt(mse),r2))
# We perform the randomforest regression 

regressor_rf=RandomForestRegressor(random_state=0)



regressor_rf.fit(X_train,y_train)



y_pred_rf=regressor_rf.predict(X_test)
# We check the accuracy of the model



mse=mean_squared_error(y_pred_rf,y_test)



r2=r2_score(y_pred_rf,y_test)



print('The Mean Squared error is {}\nThe r2 Score is {}'.format(np.sqrt(mse),r2))
# We perfom the gradient boosting regressor



regressor_gb=GradientBoostingRegressor(random_state=0)



regressor_gb.fit(X_train,y_train)



y_pred_gb=regressor_gb.predict(X_test)
# We check the accuracy of the model



mse=mean_squared_error(y_pred_gb,y_test)



r2=r2_score(y_pred_gb,y_test)



print('The Mean Squared error is {}\nThe r2 Score is {}'.format(np.sqrt(mse),r2))
# We perform the Extra Tree Regression



regressor_er=ExtraTreesRegressor()



regressor_er.fit(X_train,y_train)



y_pred_er=regressor_er.predict(X_test)
# We check the accuracy of the model



mse=mean_squared_error(y_pred_er,y_test)



r2=r2_score(y_pred_er,y_test)



print('The Mean Squared error is {}\nThe r2 Score is {}'.format(np.sqrt(mse),r2))