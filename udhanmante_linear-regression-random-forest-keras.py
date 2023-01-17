#importing necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/train.csv')
df.head()
df.info()
#overview of correlation between features as well as features Vs target
plt.subplots(figsize=(20,10))
sns.heatmap(df.corr())
#Finding out null columns entries density wise
plt.subplots(figsize=(20,10))
sns.heatmap(df.isnull())
 #These are the columns with most of null entries
df[['Id','LotFrontage','Alley','FireplaceQu','PoolQC','Fence','MiscFeature']]
#Removing some of nearly null features(Columns) along with Id column as it has no resemblance with SalePrice of House
#Note: We didnt delete LotFrontage column because less sparcity and we can process the column to fill null values as it is float
df.drop(['Id','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
plt.subplots(figsize=(20,10))
sns.heatmap(df.isnull())
#Filling mean value of LotFrontage column into empty rows of the column
df['LotFrontage'].mean()
def averaging(value):
    if pd.isnull(value):
        return 70.04995836802665
    else:
        return value
df['LotFrontage'] = df['LotFrontage'].apply(averaging)
#again Visualizing null plot
plt.subplots(figsize=(20,10))
sns.heatmap(df.isnull())
#Removing remaining null entries so as to standardize Data
df.dropna(inplace=True)
plt.subplots(figsize=(20,10))
sns.heatmap(df.isnull())
#Taking categorical columns togather intoseparate dataframe
df_catrgories = pd.concat([df.select_dtypes(include='object'),df[['MSSubClass','YrSold']]],axis=1)
df_catrgories.head()
#Converting categorical columns into dummy variables
df_catFeatures = pd.get_dummies(df_catrgories,drop_first=True)
df.drop(df_catrgories.columns,axis=1,inplace=True)
#Creating a final Dataframe with all features as numbers
df_final = pd.concat([df_catFeatures,df],axis=1)
#Splitting final Dataframe into features and Target
features = df_final.drop('SalePrice',axis=1)
Target = df_final['SalePrice']
features.info()
#Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(features)
scaled_features = scaler.transform(features)
scaled_features
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(scaled_features,Target)
pca_features = pca.transform(scaled_features)
pca_features.shape
type(pca_features)
pca_feature_df = pd.DataFrame(pca_features,columns=['A','B','C','D','E','F','G','H','I','J'])
pca_feature_df.head()
pca_df = pd.concat([pca_feature_df,df['SalePrice']],axis=1)
pca_df.head()
plt.subplots(figsize=(20,10))
sns.heatmap(pca_df.corr())
#Obtaining scatter plot for all 10 principle components with target
fig,ax = plt.subplots(figsize=(20,10))
for column in pca_df.drop('SalePrice',axis=1).columns:
    plt.scatter(pca_df[column],pca_df['SalePrice'])
    plt.xlabel(column)
    plt.ylabel('SalePrice')
    plt.legend()
for column in pca_df.drop('SalePrice',axis=1).columns:
    fig,ax = plt.subplots(figsize=(10,6))
    plt.scatter(pca_df[column],pca_df['SalePrice'])
    plt.xlabel(column)
    plt.ylabel('SalePrice')
    plt.legend()
#Splitting our Data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, Target, test_size=0.3, random_state=42)
#Linear Regression Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
lm_predictions = lm.predict(X_test)
#Checking variance of predicted and true values by scatter plot
plt.scatter(lm_predictions,y_test)
plt.title('Regression Using Linear regression model in sklearn')
plt.xlabel('lm_prediction')
plt.ylabel('y_test')
#Random Forest Regression Model
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=10)
rfr.fit(X_train,y_train)
rfr_prediction = rfr.predict(X_test)
#Checking variance of predicted and true values by scatter plot
plt.scatter(rfr_prediction,y_test)
plt.title('Regression Random Forest Regressor')
plt.xlabel('rfr_prediction')
plt.ylabel('y_test')
#Keras Neural network model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam
model = Sequential()
model.add(Dense(227,input_shape=(227,),activation='relu'))
model.add(Dense(227,activation='relu'))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X_train,y_train,epochs=40,validation_split=0.1)
keras_prediction = model.predict(X_test)
#Checking variance of predicted and true values by scatter plot
plt.scatter(keras_prediction,y_test)
plt.title('Regression Using Keras Neural Network')
plt.xlabel('keras_prediction')
plt.ylabel('y_test')
#Finally Comparing result from three models used above
#Checking variance of predicted and true values by scatter plot Linear Regression
plt.subplots(figsize=(12,6))
plt.scatter(lm_predictions,y_test)
plt.title('Regression Using Linear regression model in sklearn')
plt.xlabel('lm_prediction')
plt.ylabel('y_test')

#Checking variance of predicted and true values by scatter plot Random Forest Regression
plt.subplots(figsize=(12,6))
plt.scatter(rfr_prediction,y_test)
plt.title('Regression Random Forest Regressor')
plt.xlabel('rfr_prediction')
plt.ylabel('y_test')

#Checking variance of predicted and true values by scatter plot Neural Network
plt.subplots(figsize=(12,6))
plt.scatter(keras_prediction,y_test)
plt.title('Regression Using Keras Neural Network')
plt.xlabel('keras_prediction')
plt.ylabel('y_test')
#Conclusion: In this case the Linear Regression Model predict more accurate than the other two.