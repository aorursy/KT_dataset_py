import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
sns.set(style='whitegrid')
train=pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
test=pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')
print("Training data shape: ", train.shape) # 381109 rows, 12 columns
print("Test data shape: ", test.shape)   # 127037 rows, 11 columns (missing the response column deliberately)
train.head() # starts at id #1
test.head() # starts at id #381110
train.isnull().sum() # no null values
train.dtypes
for column in ['Region_Code','Policy_Sales_Channel']:
    train[column] = train[column].astype('int')
    test[column] = test[column].astype('int')
numerical_columns=['Age', 'Region_Code','Annual_Premium','Vintage']
categorical_columns=['Gender','Driving_License','Previously_Insured','Vehicle_Age','Vehicle_Damage','Response']
train[numerical_columns].describe()
for category in categorical_columns:
    print(train[category].value_counts(), '\n______________________\n')
sns.countplot(train.Response)
train.Response.value_counts()
train.Response.value_counts()[1]/(train.Response.value_counts()[1]+train.Response.value_counts()[0])
print("Total age distribution:\n\n",
      train.Age.describe(),
      "\n_______________________\n\n",
      "Age distribution where Response = 1:\n\n",
      train.Age.loc[train.Response == 1].describe())
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(train.Age, label = "Total customers", bins=65)
sns.distplot(train.Age.loc[train.Response == 1], label = "Customers who purchased vehicle insurance", bins=64)
plt.legend()
# There are 22 people over the age of 83, and none of them wanted vehicle insurance.  Maybe it would be too expensive for them.
# The model will likely predict a response of 0 for the very old as well as the very young.
train.loc[train.Age >83]
sns.distplot(train.Annual_Premium, label = "Total customers")
sns.distplot(train.Annual_Premium.loc[train.Response == 1], label = "Customers who purchased vehicle insurance")
plt.legend()
sns.scatterplot(x=train['Age'],y=train['Annual_Premium']) # There does not appear to be much correlation between age and annual premium
df=train.groupby(['Gender','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
g = sns.catplot(x="Gender", y="count",col="Response",
                data=df, kind="bar",
                height=4, aspect=.7);
df=train.groupby(['Previously_Insured','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
g = sns.catplot(x="Previously_Insured", y="count",col="Response",
                data=df, kind="bar",
                height=4, aspect=.7);
df
train.groupby(['Previously_Insured','Response'])['id'].count()
df=train.groupby(['Vehicle_Age','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
df
g = sns.catplot(x="Vehicle_Age", y="count",col="Response",
                data=df, kind="bar",
                height=4, aspect=.7);
df=train.groupby(['Vehicle_Damage','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()
df
df2=pd.DataFrame({'total': train['Vehicle_Damage'].value_counts(), 'Response=1':train.loc[train['Response'] == 1,'Vehicle_Damage'].value_counts()})
df2['Response Rate'] = df2['Response=1']/df2['total']
df2
g = sns.catplot(x="Vehicle_Damage", y="count",col="Response",
                data=df, kind="bar",
                height=4, aspect=.7);
sns.distplot(train.Vintage, label = "Total customers", bins = 30)
sns.distplot(train.Vintage.loc[train.Response == 1], label = "Customers who purchased vehicle insurance", bins=30)
plt.legend() # the slight sawtooth pattern is just a result of binning, nothing else
df=pd.DataFrame({'total': train['Region_Code'].value_counts(), 'Response=1':train.loc[train['Response'] == 1,'Region_Code'].value_counts()})
df['Response Rate'] = df['Response=1']/df['total']
df.sort_values('Response Rate')
#As we can see, different regions have very different response rates, ranging from about 4% to 19%.  Those regions however were smaller samples than other ones, and may be outliers.
df=pd.DataFrame({'total': train['Policy_Sales_Channel'].value_counts(), 'Response=1':train.loc[train['Response'] == 1,'Policy_Sales_Channel'].value_counts()})
df['Response Rate'] = df['Response=1']/df['total']
df.sort_values('Response Rate')
df.loc[df['Response Rate'].isnull(),:]
df.sort_values('Response Rate').dropna()
# Response rates by sales channel vary from 2 to 100%, though some of these are outliers due to small samples.  Channel 152 is noteworthy because it has almost
# 135,000 samples, and its response rate is only 2.86%.
df.sort_values('total', ascending = False).dropna().iloc[:20] # Take the 20 top sales channels by number of customers.  We don't care about the puny ones.
# The top 5 channels make up about 300,000 of 382,000 total customers, and they vary widely in response rate from about 2% to 20%.  Clearly this is important information
num_feat = ['Age','Vintage']
cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year','Vehicle_Age_gt_2_Years','Vehicle_Damage_Yes','Region_Code','Policy_Sales_Channel']
train['Gender'] = train['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
train=pd.get_dummies(train,drop_first=True)
train=train.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
train['Vehicle_Age_lt_1_Year']=train['Vehicle_Age_lt_1_Year'].astype('int')
train['Vehicle_Age_gt_2_Years']=train['Vehicle_Age_gt_2_Years'].astype('int')
train['Vehicle_Damage_Yes']=train['Vehicle_Damage_Yes'].astype('int')
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
ss = StandardScaler()
train[num_feat] = ss.fit_transform(train[num_feat])


mm = MinMaxScaler()
train[['Annual_Premium']] = mm.fit_transform(train[['Annual_Premium']])
train=train.drop('id',axis=1) # drop the id column

for column in cat_feat:
    train[column] = train[column].astype('str')
    
test['Gender'] = test['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
test=pd.get_dummies(test,drop_first=True)
test=test.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
test['Vehicle_Age_lt_1_Year']=test['Vehicle_Age_lt_1_Year'].astype('int')
test['Vehicle_Age_gt_2_Years']=test['Vehicle_Age_gt_2_Years'].astype('int')
test['Vehicle_Damage_Yes']=test['Vehicle_Damage_Yes'].astype('int')
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
ss = StandardScaler()
test[num_feat] = ss.fit_transform(test[num_feat])


mm = MinMaxScaler()
test[['Annual_Premium']] = mm.fit_transform(test[['Annual_Premium']])
for column in cat_feat:
    test[column] = test[column].astype('str')
from sklearn.model_selection import train_test_split

train_target=train['Response']
train=train.drop(['Response'], axis = 1)
x_train,x_test,y_train,y_test = train_test_split(train,train_target, random_state = 0)
id=test.id
test=test.drop('id',axis=1)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
#from catboost import CatBoostClassifier
#from scipy.stats import randint
import pickle
#import xgboost as xgb
#import lightgbm as lgb
from sklearn.metrics import accuracy_score
# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score,accuracy_score,confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, classification_report 
%pylab inline
x_train.dtypes
random_search = {'criterion': ['entropy', 'gini'],
               'max_depth': [2,3,4,5,6,7,10],
               'min_samples_leaf': [4, 6, 8],
               'min_samples_split': [5, 7,10],
               'n_estimators': [300]}

clf = RandomForestClassifier()
model = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 10, 
                               cv = 4, verbose= 1, random_state= 101, n_jobs = -1)
model.fit(x_train,y_train)
filename = 'rf_model.sav'
pickle.dump(model, open(filename, 'wb'))
rf_load = pickle.load(open(filename, 'rb'))
rf_load
y_pred=model.predict(x_test)
y_pred[:300] # this model appears to predict 0 for everything
np.unique(y_pred, return_counts=True) # Yes, it predicts 0 for everything in this case (your result may be a bit different)
list(y_pred).count(0)
np.sort(model.predict_proba(x_test)[:,1]) # this function yields a 2d array with the probability of "0" in the 0th column and of "1" in the 1st column. Hence the [:,1]
sns.distplot(model.predict_proba(x_test)[:,1])
print (classification_report(y_test, y_pred)) # By predicting 0 for everybody, the model achieves about 88% accuracy
y_score = model.predict_proba(x_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)  

title('Random Forest ROC curve: Insurance Purchse')
xlabel('FPR (Precision)') # false positive rate
ylabel('TPR (Recall)')   # true positive rate

plot(fpr,tpr)
plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr)) # about 0.855
sns.distplot(y_score) # same plot from earlier
score = roc_auc_score(y_test, y_score)
print("auc-roc score on Test data",score) # 0.855
train=pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
test=pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')

train['Gender'] = train['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
test['Gender'] = test['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)

for column in ['Region_Code','Policy_Sales_Channel']:
    train[column] = train[column].astype('int')
    test[column] = test[column].astype('int')
    
id=test.id  # capture id for later, test id only, for final submission
    
train=train.drop('id',axis=1) # drop the id column
test=test.drop('id',axis=1)

cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',
 'Region_Code', 'Policy_Sales_Channel']

for column in cat_feat:
    train[column] = train[column].astype('str')

for column in cat_feat:
    test[column] = test[column].astype('str')

train['Age'] = train['Age']//5 # divide all ages by 5 to get them into bins of 5 years each for the dummy variables
train['Age'] = train['Age'].astype(str)

test['Age'] = test['Age']//5
test['Age'] = test['Age'].astype(str)

train['Annual_Premium'] = ((train['Annual_Premium'])//1000)**0.5//1 #bin the annual premium into about 20 bins, with smaller bin sizes for smaller amounts
train['Annual_Premium'] = train['Annual_Premium'].astype(str)
test['Annual_Premium'] = ((test['Annual_Premium'])//1000)**0.5//1
test['Annual_Premium'] = test['Annual_Premium'].astype(str)

train=pd.get_dummies(train,drop_first=True)
test=pd.get_dummies(test,drop_first=True)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

mm = MinMaxScaler()
train[['Vintage']] = mm.fit_transform(train[['Vintage']])
test[['Vintage']] = mm.fit_transform(test[['Vintage']])  # This simply reduces Vintage to smaller numbers.  They are not expected to make a difference to the model because the distribution is flat
print(train.shape,
      test.shape) 

#(381109, 228)
#(127037, 217)
rejectColumns = []
i = 0
for name in list(train.columns):
    if name not in list(test.columns):
        print(name)
        rejectColumns.append(name)
print(i)
print(rejectColumns)
rejectColumns.remove('Response')
rejectColumns
for name in train.columns:
    #print(name)
    if name in rejectColumns:
        print(name)
        train = train.drop(name,axis = 1)
i = 0
rejectColumns = []
for name in list(test.columns):
    if name not in list(train.columns):
        print(name)
        rejectColumns.append(name)
        i += 1
print(i)
print(rejectColumns)
for name in test.columns:
    #print(name)
    if name in rejectColumns:
        print(name)
        test = test.drop(name,axis = 1)
print(train.shape,
      test.shape) 

#(381109, 216)
#(127037, 215)
from sklearn.model_selection import train_test_split
train_target=train['Response']
train=train.drop(['Response'], axis = 1)
x_train,x_test,y_train,y_test = train_test_split(train,train_target, random_state = 0)
y_test.value_counts()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# Deep Learning Libraries
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
y_train
y_train2 = np.array([y_train,1-y_train])
y_train2 = np.transpose(y_train2)
y_train2

y_test2 = np.array([y_test,1-y_test])
y_test2 = np.transpose(y_test2)
y_test2
y_test2[:15]
ratio = y_train.value_counts()[1]/len(y_train)
EPOCHS = 5
BATCH_SIZE = 128

#n_cols = predictors.shape[1]
early_stopping_monitor = EarlyStopping(patience=2)
class_weight = {0:1-ratio, 1:ratio-0.1}#{0:ratio, 1:1-ratio}

model = Sequential()
model.add(Dense(24, activation='relu', input_shape = (x_train.shape[1],)))
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train2, batch_size = BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, #callbacks = [early_stopping_monitor],
          class_weight=class_weight)
# bs320 ep 100, ROC 0.845
# bs32 ep 10, ROC 0.855
# bs32 ep 10, val 0.2, ROC 0.8556
# bs32 ep 10, val 0.1, ROC 0.8553
# bs32 ep 10, val 0.3, ROC 0.8547
# bs32 ep 10, val 0.2, ROC 0.8549
# bs3200 ep 100, val 0.2, ROC 0.851
# bs3200 ep 100, val 0.2, ROC 0.845
# bs16 ep 3, val 0.2, ROC 0.8546
# bs32 ep 3, val 0.2, ROC 0.8558
# bs64 ep 3, val 0.2, ROC 0.8559
# bs64 ep 3, val 0.2, ROC 0.8550
# bs32 ep 3, val 0.2, ROC 0.8557
# bs16 ep 3, val 0.2, ROC 0.8552
# bs32 ep 3, val 0.2, ROC 0.8548
# bs32 ep 5, val 0.2, ROC 0.8544
# bs32 ep 5, val 0.2, ROC 0.8545
# bs16 ep 5, val 0.2, ROC 0.8543
# bs64 ep 5, val 0.2, ROC 0.8549
# bs128 ep 5, val 0.2, ROC 0.8549
# bs128 ep 5, val 0.1, ROC 0.8551
# bs128 ep 5, val 0.3, ROC 0.8548
# bs128 ep 5, val 0.05, ROC 0.8541


# for use in Evaluation Notebook
# model.save('./tfModelInsurance.h5') 
# x_train.to_csv('./x_train.csv')
model.get_weights() #6 / 234[24, 24, 24 ...], 24, 24[20, 20 ...], 20, 20[2, 2 ...], 2: [ 0.02806583, -0.02806598]
model.get_weights()[0][2]
layer1wtsum = []
for i in range(234):
    layer1wtsum.append(sum(model.get_weights()[0][i]))
layer1wtsumdf = pd.DataFrame({'column': train.columns, 'weightsum': layer1wtsum})
layer1wtsumdf[:20]
layer1wtsumdf.sort_values(by='weightsum')[-15:]
print("Evaluate on test data")
results = model.evaluate(x_test, y_test2, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:\n", predictions)
y_test3 = y_test2
y_testpred = model.predict(x_test)[:,0]
offset = 0.1 #distance from 0.5 to serve as cutoff for prediction in probabilities
y_test3[:,1] = np.around(y_testpred-offset)
# 0.2 -> 23.8% positive, 0.15 -> 28.9%, 0.1 -> 32.8%, 
tp = 0
fp = 0
fn = 0
tn = 0

for i in range(len(y_test3)):
    if all(y_test3[i] == [1,1]):
        tp += 1
    elif all(y_test3[i] == [0,1]):
        fp += 1
    elif all(y_test3[i] == [1,0]):
        fn += 1
    elif all(y_test3[i] == [0,0]):
        tn += 1
        
print(tp, fp, fn, tn, tp+fp+fn+tn, len(y_test3))
cm = np.array([[tp, fn],[fp, tn]]) # confusion matrix
cm
y_test.value_counts()[1]/len(y_test)
(tp+fp)/(len(y_test3)) # how many things are being predicted to be positive.  It appears to be too many positive predictions
# This step is not strictly necessary.  The final submission will assign each test id a probability between 0 and 1, not a prediction of exactly 0 or 1
# This was just an exercise
from sklearn.metrics import f1_score, roc_auc_score,accuracy_score,confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, classification_report 
import matplotlib.pyplot as plt
y_score = model.predict(x_test)[:,0]
fpr, tpr, _ = roc_curve(y_test, y_score)

plt.title('Dense Neural Net ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr))
# yields AUC of 0.855, very similar to random forest
train_target2 = np.array([train_target,1-train_target])
train_target2 = np.transpose(train_target2)
train_target2
ratio = np.unique(train_target2[:,0], return_counts=True)[1][1]/len(train_target2)
EPOCHS = 10
BATCH_SIZE = 128

#n_cols = predictors.shape[1]
early_stopping_monitor = EarlyStopping(patience=2)
class_weight = {0:1-ratio, 1:ratio-0.1}#{0:ratio, 1:1-ratio}

model = Sequential()
model.add(Dense(25, activation='relu', input_shape = (train.shape[1],)))
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train, train_target2, batch_size = BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, #callbacks = [early_stopping_monitor],
          class_weight=class_weight)
score = model.predict(test)[:,0]
sns.distplot(score)
y_score = model.predict(train)[:,0]
fpr, tpr, _ = roc_curve(train_target, y_score)

plt.title('Dense Neural Net ROC curve')
plt.xlabel('FPR (Precision)')
plt.ylabel('TPR (Recall)')

plt.plot(fpr,tpr)
plt.plot((0,1), ls='dashed',color='black')
plt.show()
print ('Area under curve (AUC): ', auc(fpr,tpr))
# The AUC is 0.862 for the training data.  The test AUC is expected to be 0.855 as it was for the previous ROC where the model had not seen the test data.  
# The difference between 0.862 and 0.855 is small.  This is a good sign.  If overfitting had occurred, it would be much larger.
submission = pd.DataFrame(data = {'id': id, 'Response': score})
submission.to_csv('vehicle_insurance_tensorflow1.csv', index = False)
submission.head()