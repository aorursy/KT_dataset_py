import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
train=pd.read_csv("../input/application_train.csv")
test=pd.read_csv("../input/application_test.csv")
train.head()
(train['DAYS_BIRTH']/-365).describe()
# Missing Value Function
def missing_values(df):
    mis_val=df.isnull().sum() #Total missing values
    mis_percent=100*(df.isnull().sum()/len(df)) #Percentage of missing values
    mis_table=pd.concat([mis_val, mis_percent], axis=1) #Table of missing values
    mis_rename_col=mis_table.rename(columns= {0: 'Missing Values', 1: 'Percent'}) #Rename columns
    mis_rename_col=mis_rename_col[mis_rename_col.iloc[:,1]!=0].sort_values('Percent', ascending=False).round(1)#Sort table desc
    return mis_rename_col #return df with missing info
miss_values=missing_values(train)
miss_values.head(20)
train.select_dtypes('object').apply(pd.Series.nunique, axis=0)
le=LabelEncoder()#label encoder object
le_count=0

# Apply label encoding for categorical variables with only 2 categories
for col in train:
    if train[col].dtype=='object':
        if len(list(train[col].unique())) <=2:
            le.fit(train[col])
            train[col]=le.transform(train[col])
            test[col]=le.transform(test[col])
            le_count += 1
#Apply One hot encoding for categorical variables with more than 2 categories
train=pd.get_dummies(train)
test=pd.get_dummies(test)
train_labels=train['TARGET']
train, test= train.align (test, join= 'inner', axis=1)# Align both data and keep only common columns from both dataframes
train['TARGET']=train_labels
train['TARGET'].plot.hist()
train['DAYS_EMPLOYED'].plot.hist(title= 'Days Employment Histogram')
plt.xlabel('Days Employment')
train['DAYS_EMPLOYED'].describe()
anom= train[train['DAYS_EMPLOYED']==365243]
non_anom=train[train['DAYS_EMPLOYED']!=365243]
print('Non-anomalies default on %0.2f%% of loans' % (100* non_anom['TARGET'].mean()))
print('There are %d anomalies of days employed' % len(anom))
train['DAYS_EMPLOYED_ANOM']=train['DAYS_EMPLOYED']==365243 #Flag for anomalous column
train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace= True) # Anomalous replaced with nan
train['DAYS_EMPLOYED'].plot.hist(title= 'Days Employment Histogram')
plt.xlabel('Days Employed')
test['DAYS_EMPLOYED_ANOM']=test['DAYS_EMPLOYED']==365243
test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
print('There are %d an0malies in test data out of %d entries' % (test['DAYS_EMPLOYED_ANOM'].sum(), len(test)))
correlation=train.corr()['TARGET'].sort_values() # Correlations with target
print('Ten most positive correlation: \n', correlation.tail(10))
print('Ten most negative correlation: \n', correlation.head(10))
train['DAYS_BIRTH']=abs(train['DAYS_BIRTH'])
train['DAYS_BIRTH'].corr(train['TARGET'])
plt.style.use('fivethirtyeight') #plot style
plt.hist(train['DAYS_BIRTH']/365, edgecolor= 'k', bins = 25) #Dist of age in years
plt.title('Age of Client')
plt.xlabel('Age (Yr)'); plt.ylabel('Count')
plt.figure(figsize =  (10, 8))
sns.kdeplot(train.loc[train['TARGET']==0, 'DAYS_BIRTH']/365, label='target==0') #KDE plot of loans that were repaid on time
sns.kdeplot(train.loc[train['TARGET']==1, 'DAYS_BIRTH']/365, label= 'targer==1') #KDE plot of loans that were not repaid on time
plt.xlabel('Age (Yr)'); plt.ylabel('Density');plt.title('Dist of Age')
age_data=train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH']=age_data['DAYS_BIRTH']/ 365 #Age as a new data frame
age_data['YEARS_BINNED']=pd.cut(age_data['YEARS_BIRTH'], bins=np.linspace(20, 70, num = 11))
age_data.head(10)
age_groups= age_data.groupby('YEARS_BINNED').mean() #group by bins with averages
age_groups
plt.figure(figsize= (8, 8))
plt.bar(age_groups.index.astype(str), 100*age_groups['TARGET']) #Bar plot of target and age in bins
plt.xticks(rotation=75);plt.xlabel('Age Group (Yrs)');  plt.ylabel('Failure to repay (%)'); plt.title('Failure to Repay by Age Group')
#Extract the ext_source variables and show correlations
ext_data=train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_corrs=ext_data.corr()
ext_corrs
plt.figure(figsize= (8, 6))
sns.heatmap(ext_corrs, cmap=plt.cm.RdYlBu_r, vmin= -0.25, annot= True, vmax= 0.6)
plt.title('Correlation Heatmap')
plt.figure(figsize= (10,12))
for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    plt.subplot(3,1,i+1)
    sns.kdeplot(train.loc[train['TARGET']==0, source], label= 'target==0')
    sns.kdeplot(train.loc[train['TARGET']==1, source], label= 'target==1')
    plt.title("Dist of %s by Target Value"% source); plt.xlabel('%s'% source); plt.ylabel('Density');
plt.tight_layout(h_pad=2.5)
plot_data=ext_data.drop(columns=['DAYS_BIRTH']).copy()
plot_data['YEARS_BIRTH']=age_data['YEARS_BIRTH']
plot_data=plot_data.dropna().loc[:10000, :]
def corr_func(x,y, **kwargs):
    r= np.corrcoef(x,y)[0][1]
    ax=plt.gca()
    ax.annotate("r= {:.2f}".format(r),
               xy=(.2, .8), xycoords=ax.transAxes,
               size = 20)
grid = sns.PairGrid(data=plot_data, size= 3, diag_sharey=False,
                   hue= 'TARGET', vars= [x for x in list(plot_data.columns) if x != 'TARGET'])
grid.map_upper(plt.scatter, alpha=0.2)
grid.map_diag(sns.kdeplot)
grid.map_lower(sns.kdeplot, cmap= plt.cm.OrRd_r)
plt.suptitle('Ext Source and Age Features Pairs Plot', size= 32, y= 1.05)
poly_features=train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test=test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

from sklearn.preprocessing import Imputer
imputer= Imputer(strategy='median')
poly_target=poly_features['TARGET']
poly_features=poly_features.drop(columns= ['TARGET'])

poly_features= imputer.fit_transform(poly_features)
poly_features_test=imputer.transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures

poly_transformer= PolynomialFeatures(degree= 3)

poly_transformer.fit(poly_features)
poly_features= poly_transformer.transform(poly_features)
poly_features_test= poly_transformer.transform(poly_features_test)
poly_features=pd.DataFrame(poly_features, columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))
poly_features_test=pd.DataFrame(poly_features_test, columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))
poly_features['TARGET']=poly_target
poly_corrs=poly_features.corr()['TARGET'].sort_values()
print(poly_corrs.head(10))
print(poly_corrs.tail(5))
#Merging polynomial features into train & test data
poly_features['SK_ID_CURR']=train['SK_ID_CURR']
train_poly=train.merge(poly_features, on=  'SK_ID_CURR', how= 'left')
poly_features_test['SK_ID_CURR']=train['SK_ID_CURR']
test_poly=train.merge(poly_features_test, on=  'SK_ID_CURR', how= 'left')
train_poly, test_poly=train_poly.align(test_poly, join='inner', axis=1)
from sklearn.preprocessing import MinMaxScaler, Imputer

if 'TARGET' in train:
    train1=train.drop(columns=['TARGET'])
else:
    train1= train.copy()

features=list(train.columns)
test1=test.copy()
# Imputation
imputer= Imputer(strategy='median')
scaler= MinMaxScaler(feature_range=(0, 1))
imputer.fit(train1)
train1= imputer.transform(train1)
test1= imputer.transform(test)

scaler.fit(train1)
train1=scaler.transform(train1)
test1=scaler.transform(test1)
from sklearn.linear_model import LogisticRegression

log_reg= LogisticRegression(C=0.0001)
log_reg.fit(train1, train_labels)
log_reg_pred=log_reg.predict_proba(test1)[:,1]
submit = test[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

submit.head()
submit.to_csv('log_reg.csv', index= False)