#loading need libraries

import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats

%matplotlib inline



import seaborn as sns



from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats



# Evaluating matrices

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("train.csv") # reading the traing csv file

df.tail() # verifying the below five records of the train dataset
df.info() # attribute visualization of train dataset
df.isnull().sum() #checking the null values interm of True and false

# or can check null values interms of count (i.e, df.isnull().sum())
z = df.isnull().any()

df.columns[z] # checking by column names
#plot the missing value attributes

plt.figure(figsize=(12, 6))

sns.heatmap(df.isnull())

plt.show()
df.isnull().sum()#.max()

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)



missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(15)
# Separating numerical values and categorical value

numerical   = df.select_dtypes(exclude=['object'])

categorical = df.select_dtypes(include=['object'])

df1= numerical # working on categorical value

df1
del numerical['Id'] # respresnts the ID but not useful to train attribute to train it
# Findimg the correlation matrix for the categorical features





corrmat = numerical.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(numerical[top_corr_features].corr(),annot=True,cmap="RdYlGn")
correlation = numerical.corr()

correlation_matrix_top_features = numerical.corr().abs()



# Select upper triangle of correlation matrix

upper = correlation_matrix_top_features.where(np.triu(np.ones(correlation_matrix_top_features.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.5

top_feature = [column for column in upper.columns if any(upper[column] > 0.5)]

top_corr = df[top_feature].corr()



print("+++++++++++++++++Find most important features relative to target+++++++++++++++++++")

corr = df.corr()

corr.sort_values(['SalePrice'], ascending=False, inplace=True)

corr.SalePrice
# Reverifying the missing values in the categorical features



df1.isnull().sum()#.max()

total = df1.isnull().sum().sort_values(ascending=False)

percent = (df1.isnull().sum()/df1.isnull().count()).sort_values(ascending=False)



missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(15)
step1 = missing_data[(missing_data['Percent'] > .20)]

df1 = df1.drop(step1.index, 1)
step2 = missing_data[(missing_data['Percent'] <= .20) & (missing_data['Percent'] > 0.10)]

df1 = df1.dropna(subset=step2.index)
step3 = missing_data[(missing_data['Percent'] <= .10) & (missing_data['Percent'] > 0)]

step3
df1[step3.index] = df1[step3.index].fillna(df1.mode().iloc[0])

df1.isnull().sum().max()
plt.figure(figsize=(10, 5))

sns.heatmap(df1.isnull())
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
def preprocess(df1, str_labels):

    for label in str_labels:

        le.fit(df1[label])

        df1[label] = le.transform(df1[label])

    return df1



df1 = preprocess(df1, df1.columns)
df1
# Applying normalization technique



normalized = preprocessing.normalize(df1[0:37])

normalized
y = df1['SalePrice'].values

X = df1.loc[:, df1.columns != 'SalePrice'].values

y.shape


# Split data into train and test formate

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)




#Train the model

from sklearn import linear_model

model = linear_model.LinearRegression()



model.fit(X_train, y_train)
#Prediction

print("Predict value " + str(model.predict([X_test[142]])))

print("Real value " + str(y_test[142]))
print("Accuracy --> ", model.score(X_test, y_test)*100)
#Train the model

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100,random_state = 1)

model.fit(X_train, y_train)

preds= model.predict(X_test)
print("Accuracy --> ", model.score(X_test, y_test)*100)
#Train the model

from sklearn.ensemble import GradientBoostingRegressor

G_Regressor = GradientBoostingRegressor(n_estimators=250, max_depth=3)
## Fit

G_Regressor.fit(X_train, y_train)
print("Accuracy --> ", G_Regressor.score(X_test, y_test)*100)
df2= pd.read_csv("test.csv") # importing the test data 

df2
# Separating numerical values and categorical value

numerical   = df2.select_dtypes(exclude=['object'])

categorical = df2.select_dtypes(include=['object'])

df2= numerical # working on categorical value

# del numerical['Id'] # respresnts the ID but not useful to train attribute to train it



#  Findimg the correlation matrix for the numerical features





corrmat = numerical.corr()



correlation_matrix_top_features = numerical.corr().abs()



# Select upper triangle of correlation matrix

upper = correlation_matrix_top_features.where(np.triu(np.ones(correlation_matrix_top_features.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.5

top_feature = [column for column in upper.columns if any(upper[column] > 0.5)]

top_corr = df[top_feature].corr()



step1 = missing_data[(missing_data['Percent'] > .20)]

df2 = df2.drop(step1.index, 1)

step2 = missing_data[(missing_data['Percent'] <= .20) & (missing_data['Percent'] > 0.10)]

df2 = df2.dropna(subset=step2.index)



# Reverifying the missing values in the categorical features



df2.isnull().sum()#.max()

total = df2.isnull().sum().sort_values(ascending=False)

percent = (df2.isnull().sum()/df2.isnull().count()).sort_values(ascending=False)



missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(15)



step3 = missing_data[(missing_data['Percent'] <= .10) & (missing_data['Percent'] > 0)]

step3

df2[step3.index] = df1[step3.index].fillna(df2.mode().iloc[0])

df2.isnull().sum().max()
# Encoding of numerical attribute values



def preprocess(df2, str_labels):

    for label in str_labels:

        le.fit(df2[label])

        df2[label] = le.transform(df2[label])

    return df2



df2 = preprocess(df2, df2.columns)
# Applying normalization techniques

normalized = preprocessing.normalize(df2[0:37]) 

normalized
test = pd.read_csv("sample_submission.csv") # Reduces the features to the shape according to numerical value of our model.

test.shape

test_y = test["SalePrice"].values

test_y.shape
y_pred = G_Regressor.predict(X_test)

# print(y_pred)

preds_df= pd.DataFrame(test, columns=['Id'])



preds_df['SalePrice']=y_pred

preds_df.head()

data = pd.read_csv("sample_submission.csv")

data.head()





preds_df.to_csv('test_Submission.csv', index=False)


