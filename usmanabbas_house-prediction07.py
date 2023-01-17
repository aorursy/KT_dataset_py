import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train_df=pd.read_csv("../input/train.csv")
test_df=pd.read_csv("../input/test.csv")
combine= pd.concat((train_df.loc[:,'MSSubClass':'SaleCondition'],
                      test_df.loc[:,'MSSubClass':'SaleCondition']))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


combine= pd.concat((train_df.loc[:,'MSSubClass':'SaleCondition'],
                      test_df.loc[:,'MSSubClass':'SaleCondition']))


combine.head()
combine.describe()

corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat);
#Heatmap defines it all we can see that our dependent variable is correlated with many variables
#Sale Price is not uniformly distributed
train_df["SalePrice"].hist()
(np.log(train_df["SalePrice"])).hist()
total_combine=combine.isnull().sum().sort_values(ascending=False)
total_combine_perc = (combine.isnull().sum()/combine.isnull().count()).sort_values(ascending=False)
missing_data_combine=pd.concat([total_combine,total_combine_perc],axis=1,keys=["Total","Percentage"])


#removing such columns which have more than 1 missing values
combine=combine.drop((missing_data_combine[missing_data_combine["Total"]>1]).index,1)

#Replacing missing values with mean
combine["TotalBsmtSF"].fillna(combine["TotalBsmtSF"].mean(),inplace=True)
combine["KitchenQual"].fillna("Ex",inplace=True)
combine["GarageArea"].fillna(combine["GarageArea"].mean(),inplace=True)
combine["GarageCars"].fillna(combine["GarageCars"].mean(),inplace=True)
combine["SaleType"].fillna("Partial",inplace=True)

X_train=combine.iloc[0:1460,:]
X_test=combine.iloc[1460:2920,:]
y_train=train_df["SalePrice"]
y_train=pd.Series.to_frame(y_train)
new_train_df=pd.concat((X_train,y_train),axis=1)

plt.scatter(train_df["GrLivArea"],np.log(train_df["SalePrice"]))
plt.xlabel("Ground Live Area")
plt.ylabel("SalePrice")

plt.scatter(train_df["TotalBsmtSF"],np.log(train_df["SalePrice"]))
plt.xlabel("Total Basement Area")
plt.ylabel("SalePrice")
plt.scatter(train_df["GarageArea"],np.log(train_df["SalePrice"]))
plt.xlabel("Garage Area")
plt.ylabel("SalePrice")
#Above graphs show that there are strong correlations of grade (ground) living area,Total Basement area and 
#garage area with SalePrice but there are some outliers so we will remove them
new_train_df=new_train_df[new_train_df["GarageArea"]<1200]
new_train_df=new_train_df[new_train_df["TotalBsmtSF"]<3000]
new_train_df=new_train_df[new_train_df["GrLivArea"]<4000]

sns.boxplot(train_df["OverallQual"],train_df["SalePrice"])
#SalePrice is poitively correlated with OverallCond
plt.hist(train_df["YearBuilt"],bins=100)
#SalePrice increases as year increases
total=train_df.isnull().sum().sort_values(ascending=False)
percentage = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total,percentage],axis=1,keys=["Total","Percentage"])
missing_data
#removing all the columns which have more than 1 missing value 
train_df = train_df.drop((missing_data[missing_data['Total'] > 1]).index,1)
train_df = train_df.drop(train_df.loc[train_df['Electrical'].isnull()].index)
train_df.isnull().sum().max()

X_train=new_train_df.iloc[:,[34,43,12,1,26,39,55,14,13]]
X_train.head()
X_test=X_test.iloc[:,[34,43,12,1,26,39,55,14,13]]
X_test.head()

y_train=np.log(new_train_df.SalePrice)
y_train.head()
sns.barplot("OverallCond","SalePrice",data=train_df)
sns.barplot("SaleCondition","SalePrice",data=train_df)
#We see that houses having overallcond >=5 have high saleprice so we assign 1 value to such house
#for the salecondition Partial houses have the highest saleprice so we assign value 1 if the salecondition is 1 else we assign 0
#def overallcond(x):
  #  if x>=5:
  #      return 1
  #  else:
   #     return 0

#X_train["OverallCond"]=X_train.OverallCond.apply(overallcond)
#X_test["OverallCond"]=X_test.OverallCond.apply(overallcond)




#def isPartial(x):
   # if x=="Partial":
      #  return 1
  #  else:
     #   return 0

#X_train["Partial"]=X_train.SaleCondition.apply(isPartial)
#X_test["Partial"]=X_test.SaleCondition.apply(isPartial)

#X_train.drop("SaleCondition",inplace=True,axis=1)
#X_test.drop("SaleCondition",inplace=True,axis=1)





#Converting categorical variables to dummies
X_train=pd.get_dummies(X_train)
X_test=pd.get_dummies(X_test)
#Applying xgboost
import xgboost as xgb

regressor = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=7200,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)

regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)
y_pred=np.exp(y_pred) #We applied log tranformation previously so np.exp() will remove the effect of log
y_pred=y_pred.reshape(-1)
y_pred

submission = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": y_pred
    })
submission.to_csv('xgboost2.csv', index=False)
#We get a rmse of 0.14632