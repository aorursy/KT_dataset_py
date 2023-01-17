####Loading useful packages

#For data manipulation
import numpy as np
import pandas as pd

#For plotting
import matplotlib.pyplot as pp
import seaborn as sns

#This just ensures that our plots appear
%matplotlib inline                   


#For surpressing warnings
import warnings
warnings.filterwarnings('ignore')
#loading in the data
train_data = pd.read_csv("../input/test23/train.csv")
test_data = pd.read_csv("../input/test22/test.csv")

#Let's take a look at our training data
train_data.head()
#Let's take a look at our testing data
test_data.head()
#What's the shape of the data like ?
print("Training data dimenstions: ",train_data.shape)
print("Testing data dimenstions: ",test_data.shape)
#Having a closer look at one of the customers.....
train_data.iloc[1,]
#Let's look at the distribution of the target variable
pp.hist(train_data.loc[:,'target'])
pp.xlabel("target")
pp.ylabel("frequency")
pp.title("Distribution of target variable")
#Let's check the data type of each of the variables
train_data.info()
#Making's lists on variables which belong to each group

categorical_variables = [train_data.columns[i] for i in range(len(train_data.columns)) if 'cat' in train_data.columns[i]]

binary_variables = [train_data.columns[i] for i in range(len(train_data.columns)) if 'bin' in train_data.columns[i]]

interval_variables = [train_data.columns[i] for i in range(len(train_data.columns)) if (train_data.loc[:,train_data.columns[i]].dtype==float and 'cat' not in train_data.columns[i] and 'bin' not in train_data.columns[i])]

ordinal_variables = [train_data.columns[i] for i in range(len(train_data.columns)) if (train_data.loc[:,train_data.columns[i]].dtype == 'int64' and 'cat' not in train_data.columns[i] and 'bin' not in train_data.columns[i])][2:]


pd.Series(train_data.isnull().sum())
#Categorical Variables
for i in categorical_variables:
    train_data.loc[:,i].value_counts(dropna=False).plot.bar()
    pp.xlabel(i)
    pp.show()
#Binary variables
for i in binary_variables:
    train_data.loc[:,i].value_counts(dropna=False).plot.bar()
    pp.xlabel(i)
    pp.show()
#Interval variables
for i in interval_variables:
    train_data.loc[:,i].value_counts(dropna=False).plot.hist()
    pp.xlabel(i)
    pp.show()
    
#Ordinal variables
for i in ordinal_variables:
    train_data.loc[:,i].value_counts(dropna=False).plot.bar()
    pp.xlabel(i)
    pp.show()
#Categorical variables
train_data.loc[:,categorical_variables].describe()
#Binary Variables
train_data.loc[:,binary_variables].describe()
#Interval variables
train_data.loc[:,interval_variables].describe()
#Ordinal variables
train_data.loc[:,ordinal_variables].describe()
#Categorial Variables
test_data.loc[:,categorical_variables].describe()
#Binary variables
test_data.loc[:,binary_variables].describe()
#Interval variables
test_data.loc[:,interval_variables].describe()
#Ordinal Variables
test_data.loc[:,ordinal_variables].describe()
#Interval Variables
for i in interval_variables:
    sns.boxplot(train_data.loc[:,i],showfliers=True)
    pp.xlabel(i)
    pp.show()
#Ordinal variables
for i in ordinal_variables:
    sns.boxplot(train_data.loc[:,i],showfliers=True)
    pp.xlabel(i)
    pp.show()
#Finding the interquartile range
Q1 = train_data.quantile(0.25)
Q3 = train_data.quantile(0.75)
IQR = Q3 - Q1
int_var = ['ps_reg_02','ps_reg_03','ps_car_12','ps_car_13','ps_car_14','ps_car_15']
od_var = ['ps_ind_14','ps_calc_04','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14']
#Separating the outliers from the non-outliers
outliers = train_data[(train_data > (Q3 + 1.5 * IQR))|(train_data < (Q1 - 1.5 * IQR))].drop(labels='target',axis=1)
outliers['target'] = train_data['target']
non_outliers = train_data[(train_data <= (Q3 + 1.5 * IQR))&(train_data >= (Q1 - 1.5 * IQR))].drop(labels='target',axis=1)
non_outliers['target'] = train_data['target']

#for ordinal variables
for i in od_var:
    print("{} outliers have ".format(i),outliers[['target',i]].dropna(axis=0).groupby('target').count().iloc[1,0]/outliers[['target',i]].dropna(axis=0).groupby('target').count().sum(axis=0)[0],"ones")
    print("{} outliers have ".format(i),outliers[['target',i]].dropna(axis=0).groupby('target').count().iloc[0,0]/outliers[['target',i]].dropna(axis=0).groupby('target').count().sum(axis=0)[0],"zeroes")
    print("{} non-outliers have ".format(i),non_outliers[['target',i]].dropna(axis=0).groupby('target').count().iloc[1,0]/non_outliers[['target',i]].dropna(axis=0).groupby('target').count().sum(axis=0)[0],"ones")
    print("{} non-outliers have ".format(i),non_outliers[['target',i]].dropna(axis=0).groupby('target').count().iloc[0,0]/non_outliers[['target',i]].dropna(axis=0).groupby('target').count().sum(axis=0)[0],"zeroes")
    print("")
    print("")
#For interval variables
for i in int_var:
    print("{} outliers have ".format(i),outliers[['target',i]].dropna(axis=0).groupby('target').count().iloc[1,0]/outliers[['target',i]].dropna(axis=0).groupby('target').count().sum(axis=0)[0],"ones")
    print("{} outliers have ".format(i),outliers[['target',i]].dropna(axis=0).groupby('target').count().iloc[0,0]/outliers[['target',i]].dropna(axis=0).groupby('target').count().sum(axis=0)[0],"zeroes")
    print("{} non-outliers have ".format(i),non_outliers[['target',i]].dropna(axis=0).groupby('target').count().iloc[1,0]/non_outliers[['target',i]].dropna(axis=0).groupby('target').count().sum(axis=0)[0],"ones")
    print("{} non-outliers have ".format(i),non_outliers[['target',i]].dropna(axis=0).groupby('target').count().iloc[0,0]/non_outliers[['target',i]].dropna(axis=0).groupby('target').count().sum(axis=0)[0],"zeroes")
    print("")
    print("")
#Let's look at the correlation of the features with the response
target_correlations = train_data.corr().iloc[:,1]
target_correlations = target_correlations.iloc[2:]
target_correlations.abs().sort_values(ascending=False)
#Checking for collinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor  

vif_train_data = train_data.drop(['target','id'],axis = 1)
colnames = train_data.columns.drop(['target','id'])
vif_table = pd.DataFrame()

for i in colnames:
    a = vif_train_data.columns.get_loc(i)
    vif = variance_inflation_factor(np.array(vif_train_data),exog_idx=a)
    vif_table.loc[0,i] = vif
vif_table.loc[:,vif_table.loc[0,:] > 5]
target_correlations.abs().sort_values(ascending=False).head(5)
#Let's put the top 5 
ft_top5 = train_data.loc[:,['target','ps_car_13','ps_car_12','ps_ind_17_bin','ps_car_07_cat','ps_reg_02']]
#Let's examine the relationship between target and our top variables
for i in range(5):
#Kernal density plot of claims that did happen
    sns.kdeplot(train_data.loc[train_data.loc[:,'target'] == 0,target_correlations.abs().sort_values(ascending=False).index[i]], label = 'target ==0')
    sns.kdeplot(train_data.loc[train_data.loc[:,'target'] == 1,target_correlations.abs().sort_values(ascending=False).index[i]], label = 'target ==1')
    pp.xlabel('ps_car_13'); pp.ylabel('Density');pp.title('Distribution of {}'.format(target_correlations.abs().sort_values(ascending=False).index[i]))
    pp.show()
#Make a new dataframe for polynomial feature
poly_features = ft_top5
poly_features = poly_features.loc[:,['ps_car_13','ps_car_12','ps_ind_17_bin','ps_car_07_cat','ps_reg_02']]
poly_features_test = ft_top5.loc[:,['ps_car_13','ps_car_12','ps_ind_17_bin','ps_car_07_cat','ps_reg_02']]

from sklearn.preprocessing import PolynomialFeatures
#train the features
poly_transformer = PolynomialFeatures(degree =3)
poly_transformer.fit(poly_features)

#Transform the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
poly_features = pd.DataFrame(poly_features, columns = poly_transformer.get_feature_names(input_features=['ps_car_13','ps_car_12','ps_ind_17_bin','ps_car_07_cat','ps_reg_02']))
poly_features['target'] = train_data.loc[:,'target']
poly_corrs = poly_features.corr()['target'].sort_values()
poly_corrs.abs().sort_values(ascending = False)