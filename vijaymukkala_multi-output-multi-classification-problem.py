import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Model libraries
import xgboost as xgb


#Other Libraries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler,MinMaxScaler
#from feature_engine import categorical_encoders as ce
from sklearn.metrics import f1_score
import datetime as dt
train = pd.read_csv('/kaggle/input/pet-adoption-dataset/train.csv')
test = pd.read_csv('/kaggle/input/pet-adoption-dataset/test.csv')
print(train.shape)
train.head()
train.info()
train.describe()
f, axes = plt.subplots(ncols=2, figsize=(12,4))
sns.countplot(train['pet_category'],ax=axes[0])
sns.countplot(train['condition'],ax=axes[1])
plt.show()
print(train.isnull().sum())
print(test.isnull().sum())
train['condition'].fillna(99,inplace = True)
print(train['condition'].value_counts())
train.groupby(['condition','pet_category']).size()
train.groupby(['condition','breed_category']).size()
#Splitting the column to give provide more features 
train['nf1_pet_id'] = train['pet_id'].str[:6]
train['nf2_pet_id'] = train['pet_id'].str[:7]
print(train.nf1_pet_id.nunique())
print(train.nf1_pet_id.value_counts())
print(train.nf2_pet_id.nunique())
print(train.nf2_pet_id.value_counts())
train.groupby(['nf1_pet_id','pet_category']).size()
train.groupby(['nf2_pet_id','pet_category']).size()
# New feature(Waiting time in Months) to find the difference between the issue date and the listing date 

train['issue_date'] = pd.to_datetime(train['issue_date']).apply(lambda x: x.date()) # removing the time values from the string columns
train['listing_date'] = pd.to_datetime(train['listing_date']).apply(lambda x: x.date()) # removing the time values from the string columns
train['Waiting_time(M)'] = (train['listing_date']- train['issue_date'])/np.timedelta64(1,'M') # difference of listing_date and issue_date in months
train['Waiting_time(M)'] = np.round(train['Waiting_time(M)'],2) # rounding of the value to 2 decimals

# Also creating new features from the issue date and listing date by converting their format into months
train['issue_date'] = pd.to_datetime(train['issue_date'])
train['issue_dt_month'] = train['issue_date'].dt.month 
train['listing_date'] = pd.to_datetime(train['listing_date'])
train['listing_dt_month'] = train['listing_date'].dt.month
train[train['Waiting_time(M)'] < 0]
train = train.drop(train[train['Waiting_time(M)'] < 0].index)
train.shape
train[train['length(m)'] == 0]['length(m)'].value_counts()
f, axes = plt.subplots(ncols=2, figsize=(12,4))
sns.distplot(train['length(m)'],ax=axes[0])
sns.boxplot(train['length(m)'],ax=axes[1])
# Converting length from 'm' to 'cm'

train['length(cm)'] = train['length(m)']*100

# Replacing values less than 10 cm with the mean value

for value in train['length(cm)']:
    if (value < 10):
        train['length(cm)'] = train['length(cm)'].replace(value, np.round(train['length(cm)'].mean()))
#Creating a new feature from lenght and height as ratio
train['ratio_len_height'] = train['length(cm)']/train['height(cm)']
train['color_type'].unique()
train['color_type1'] = train['color_type']
train['color_type2'] = train['color_type']
list = ['Brown','Blue','Black','Cream','Red','Calico','Yellow','Gray','Silver','Chocolate','Orange','Liver']
for val in train['color_type']:
    if (val.split()[0] in list) & (len(val.split())>1):     
        train['color_type1'] = train['color_type1'].replace(val,val.split()[0])
        train['color_type2'] = train['color_type2'].replace(val,val.split()[1])
test['condition'].fillna(99,inplace = True)
test['nf1_pet_id'] = test['pet_id'].str[:6]
test['nf2_pet_id'] = test['pet_id'].str[:7]
test['issue_date'] = pd.to_datetime(test['issue_date']).apply(lambda x: x.date())
test['listing_date'] = pd.to_datetime(test['listing_date']).apply(lambda x: x.date())
test['Waiting_time(M)'] = (test['listing_date']- test['issue_date'])/np.timedelta64(1,'M')
test['Waiting_time(M)'] = np.round(test['Waiting_time(M)'],2)
# Also creating new features from the issue date and listing date by converting their format into months
test['issue_date'] = pd.to_datetime(test['issue_date'])
test['issue_dt_month'] = test['issue_date'].dt.month
test['listing_date'] = pd.to_datetime(test['listing_date'])
test['listing_dt_month'] = test['listing_date'].dt.month
test = test.drop(test[test['Waiting_time(M)'] < 0].index)
test['length(cm)'] = test['length(m)']*100

# Replacing values less than 10 cm with the mean value

for value in test['length(cm)']:
    if (value < 10):
        test['length(cm)'] = test['length(cm)'].replace(value, np.round(test['length(cm)'].mean()))
test['ratio_len_height'] = test['length(cm)']/test['height(cm)']
test['color_type1'] = test['color_type']
test['color_type2'] = test['color_type']
list = ['Brown','Blue','Black','Cream','Red','Calico','Yellow','Gray','Silver','Chocolate','Orange','Liver']
for val in test['color_type']:
    if (val.split()[0] in list) & (len(val.split())>1):     
        test['color_type1'] = test['color_type1'].replace(val,val.split()[0])
        test['color_type2'] = test['color_type2'].replace(val,val.split()[1])
        
plt.subplots(figsize=(16,8))
sns.heatmap(train.corr(), annot= True)
X1 = train.drop(['breed_category','pet_category'], axis=1)
y1 = train['pet_category']
X2 = train.drop(['breed_category'], axis=1)
y2 = train['breed_category']
print(X1.shape)
print(y1.shape)
X1.head()
from sklearn.model_selection import train_test_split,cross_val_score
X_train1, X_vald1, y_train1,y_vald1 = train_test_split(X1,y1,test_size = 0.25, random_state = 42,stratify = y1)
print("Train size :" ,X_train1.shape,y_train1.shape)
print("Validation size :"  ,X_vald1.shape, y_vald1.shape)
X_train1.head()
numerical_features1 = ['Waiting_time(M)','ratio_len_height']
numerical_features2 = ['X1','X2']
categorical_features = ['condition','nf1_pet_id','issue_dt_month','listing_dt_month','color_type1','color_type2']
numeric_transformer1 = Pipeline(steps=[
    ('scaler', RobustScaler())
])

numeric_transformer2 = Pipeline(steps=[
    ('normalizer',MinMaxScaler())
])

category_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  
])

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers = [
        ('drop_columns', 'drop', ['pet_id','issue_date','listing_date','length(m)','color_type','nf2_pet_id','length(cm)','height(cm)',]),
        ('numeric1', numeric_transformer1,numerical_features1),
        ('numeric2', numeric_transformer2,numerical_features2),
        ('category', category_transformer, categorical_features)
])
preprocessor.fit(X_train1)
X_train1 = preprocessor.transform(X_train1)
X_vald1 = preprocessor.transform(X_vald1)
import xgboost as xgb
from sklearn.metrics import f1_score
xgb_model = xgb.XGBClassifier(scale_pos_weight = 0.7,
min_child_weight=  3,
learning_rate = 0.3,
gamma = 0.4,
colsample_bytree = 0.5, random_state = 42)
xgb_model.fit(X_train1, y_train1)
y_pred_xgb = xgb_model.predict(X_vald1)

s1 = f1_score(y_vald1,y_pred_xgb,average = 'weighted')
print(s1)
test1 = test
test1 = preprocessor.transform(test1)
X1 = preprocessor.transform(X1)
xgb_model = xgb.XGBClassifier(scale_pos_weight = 0.7,
min_child_weight=  3,
learning_rate = 0.3,
gamma = 0.4,
colsample_bytree = 0.5)
xgb_model.fit(X1, y1)
prediction1 = xgb_model.predict(test1)
predicted_values1 = pd.DataFrame(prediction1, columns = ['pet_category'])
print(predicted_values1)
from sklearn.model_selection import train_test_split
X_train2, X_vald2, y_train2,y_vald2 = train_test_split(X2,y2,test_size = 0.25, random_state = 42, stratify = y2)
print("Train size :" ,X_train2.shape,y_train2.shape)
print("Validation size :"  ,X_vald2.shape, y_vald2.shape)
numerical_features3 = ['Waiting_time(M)','length(cm)','height(cm)']
numerical_features4= ['X1','X2']
categorical_features1 = ['condition','color_type1','color_type2','pet_category','listing_dt_month']

numeric_transformer3 = Pipeline(steps=[
    ('scaler1', RobustScaler())
])

numeric_transformer4 = Pipeline(steps=[
    ('normalizer1',MinMaxScaler())
])

category_transformer1 = Pipeline(steps=[
    ('onehot1', OneHotEncoder(handle_unknown='ignore'))  
])


from sklearn.compose import ColumnTransformer

preprocessor1 = ColumnTransformer(
    transformers = [
        ('drop_columns', 'drop', ['pet_id','issue_date','listing_date','length(m)','color_type','nf1_pet_id','nf2_pet_id','issue_dt_month']),#,'issue_dt_month','nf2_pet_id','nf1_pet_id'
        ('numeric3', numeric_transformer3,numerical_features3),
        ('numeric4', numeric_transformer4,numerical_features4),
        ('category', category_transformer1, categorical_features1)

])
preprocessor1.fit(X_train2)
X_train2 = preprocessor1.transform(X_train2)
X_vald2 = preprocessor1.transform(X_vald2)
import xgboost as xgb
xgb_model2 = xgb.XGBClassifier(scale_pos_weight = 8,
 min_child_weight = 1,
 learning_rate = 0.01,
 gamma= 0.3,
 colsample_bytree = 0.3, 
 random_state = 42)
xgb_model2.fit(X_train2, y_train2)
y_pred_xgb2= xgb_model2.predict(X_vald2)

s2 = f1_score(y_vald2,y_pred_xgb2,average = 'weighted')
print(s2)
score = (s1+s2) * 100/2
print('final_score on the using validation data',score)
X2_1 = X2.pop('pet_category')
X2['pet_category'] = X2_1 # to have the pet_category at the last same as test data 
preprocessor1.fit(X2)
X2 = preprocessor1.transform(X2)
test2 = test
test2['pet_category'] = predicted_values1['pet_category']
test2 = preprocessor1.transform(test2)
xgb_model2 = xgb.XGBClassifier(scale_pos_weight = 8,
 min_child_weight = 1,
 learning_rate = 0.01,
 gamma= 0.3,
 colsample_bytree = 0.3)
xgb_model2.fit(X2, y2)
prediction2= xgb_model2.predict(test2)
predicted_values2 = pd.DataFrame(prediction2, columns = ['breed_category'])
print(predicted_values2)
