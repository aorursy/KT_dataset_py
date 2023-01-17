import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system management
import os

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# matplolib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

print(os.listdir("../input/"))
# Training data
app_train = pd.read_csv('../input/application_train.csv')
print('Training data shape: ', app_train.shape)
app_train.head()
# Testing data features
app_test = pd.read_csv('../input/application_test.csv')
print('Testing data shape: ', app_test.shape)
app_test.head()
app_train['TARGET'].value_counts()
app_train['TARGET'].astype(int).plot.hist();
# column별로 missing values 계산하는 함수 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
    
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df) 
    
        # Table 생성 
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1) # 행 기준 
    
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0: 'Missing values', 1: '% of Total values'})
    
        # Percentage of missing 기준으로 table 내림차순 정렬 
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total values', ascending=False).round(1) 
    
        # Summary information 출력 
        print("선택된 dataframe은 " + str(df.shape[1]) + " 개의 columns입니다.\n"
             "Missing values는 " + str(mis_val_table_ren_columns.shape[0]) + " 개의 columns에 있습니다.")
    
        return mis_val_table_ren_columns
missing_values = missing_values_table(app_train)
missing_values.head(20)
app_train.dtypes.value_counts()
# object dtype을 가지는 각 열들에 있는 unique한 class의 개수  
app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)
# label encoder 객체 생성 
le = LabelEncoder()
le_count = 0

for col in app_train:
    if app_train[col].dtype == 'object':
        # 2개 이하의 unique categories를 가지는 경우
        if len(list(app_train[col].unique())) <= 2:
            le.fit(app_train[col])
            # Training 과 test data에 적용
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            le_count += 1

print('%d개의 columns가 label encoded 되었습니다.' % le_count)
# one-hot encoding 
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training features shape: ', app_train.shape)
print('Testing features shape: ', app_test.shape)
train_labels = app_train['TARGET']

# training, testing data를 align
app_train, app_test = app_train.align(app_test, join = 'inner', axis=1)

# target 다시 넣어줌
app_train['TARGET'] = train_labels

print('Training features shape: ', app_train.shape)
print('Testing features shape: ', app_test.shape)
app_train['DAYS_BIRTH'].describe()
(app_train['DAYS_BIRTH'] / -365).describe()
app_train['DAYS_EMPLOYED'].describe()
app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');
anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]

print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
# anomalous flag column 생성
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# anomalous값을 nan으로 
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

app_train['DAYS_EMPLOYED'].plot.hist(title='Days Employment Histogram');
plt.xlabel('Days Employment');
app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"]==365243
app_test["DAYS_EMPLOYED"].replace({365243:np.nan}, inplace=True)
correlations = app_train.corr()['TARGET'].sort_values()

print('Positive Correlations:\n', correlations.tail(15))
print('\nNegative Correlations:\n', correlations.head(15))
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
app_train['DAYS_BIRTH'].corr(app_train['TARGET'])
plt.figure(figsize = (10, 8))

# KDE plot of loans that were repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET']==0, 'DAYS_BIRTH'] / 365, label='target == 0')

# KDE plot of loans that were not repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET']==1, 'DAYS_BIRTH'] / 365, label='target == 1')

# Labeling 
plt.xlabel('Age (years)');
plt.ylabel('Density');
plt.title('Distribution of Ages');