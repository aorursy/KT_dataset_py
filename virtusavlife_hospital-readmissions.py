import pandas as pd
import numpy as np
pd.options.display.max_columns = None

def capitalize_after_hyphen(x):

    a=list(x)

    a[p.index('-')+1]=a[p.index('-')+1].capitalize()

    x=''.join(a)

    return ''.join(a)

 

import pandas as pd

import requests 

#l=['patients','admdissions','diagnoses','drg-codes','icu-stays','procedures','prescriptions','d-icd-diagnoses','d-icd-procedures']

url1="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/patients?limit=50000&offset=0"

url2="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/admissions?limit=50000&offset=0"

url3="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/diagnoses?limit=50000&offset=0"

url4="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/drg-codes?limit=50000&offset=0"

url5="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/icu-stays?limit=50000&offset=0"

url6="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/procedures?limit=50000&offset=0"

url7="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/prescriptions?limit=50000&offset=0"

url8="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/d-icd-diagnoses?limit=50000&offset=0"

url9="http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/d-icd-procedures?limit=50000&offset=0"

d={}

url=[url1,url2,url3,url4,url5,url6,url7,url8,url9]

 

for x in url: 

    p = x[(x.index('v1/')+len('v1/')):x.index('?limit')]

    try:

        p=capitalize_after_hyphen(p)

    except:

        pass

    try:

        p=p[:p.index('-')]+p[p.index('-')+1:]

    except:

        pass

   

    try:

        p=capitalize_after_hyphen(p)

    except:

        pass

    try:

        p=p[:p.index('-')]+p[p.index('-')+1:]

    except:

        pass

   

    

    print(p)

   

    d['{}'.format(p)]=pd.DataFrame(requests.get(x).json()['{}'.format(p)])

    d['{}'.format(p)].to_csv('{}.csv'.format(p),encoding='utf-8', index=False)
'''
tables_req = ['admissions','patients','dIcdDiagnoses','dIcdProcedures','diagnoses','drgcodes','icuStays','procedures']
d={}
for x in tables_req:
        
        dataframe=pd.read_csv('C:/Users/utsavd/Downloads/50KData/{}.csv'.format(x))
        d["{}".format(x)]=dataframe
'''
for x in d.keys():
    d[x].drop(['row_id'],axis=1,inplace=True)
drgcodes=pd.read_csv('C:/Users/utsavd/Downloads/drgcodes.csv')
drgpatients=pd.read_csv('C:/Users/utsavd/Downloads/drgPatients.csv')
drgpatients.drop('row_id',axis=1,inplace=True)
drgcodes.drop('row_id',axis=1,inplace=True)
DrgCodes=pd.concat([d['drgcodes'],drgcodes],axis=0)
DrgCodes.drop_duplicates(keep='first')
DrgCodes=DrgCodes.sort_values(by='subject_id')
drgpatients=drgpatients[drgpatients['dod'].isnull()]
DrgCodes=DrgCodes[['subject_id','hadm_id','drg_code','description']]
DrgCodes=DrgCodes.reset_index(drop=True)
d['admissions']=d['admissions'][['subject_id','hadm_id','insurance','admission_type','diagnosis','admittime','dischtime','insurance']]
d['patients']=d['patients'][['dob','subject_id']]
df=pd.merge(d['admissions'],d['patients'],on=['subject_id'],how='left')
df['timeofstay']=pd.to_datetime(df['dischtime'])-pd.to_datetime(df['admittime'])
df['timeofstay']=df['timeofstay'].dt.days
df['age']=pd.to_datetime(df['admittime'])-pd.to_datetime(df['dob'])
df.drop('dob',axis=1,inplace=True)
df['age']=(df['age'].dt.days)/365
df['age']=df['age'].astype(int)
def is_emergency(x):
    if x=='EMERGENCY':
        return 1
    else:
        return 0
df['emergency']=df['admission_type'].apply(is_emergency)
df.admittime=pd.to_datetime(df.admittime)
df.dischtime=pd.to_datetime(df.dischtime)
df=df.sort_values(['subject_id','admittime'])
df=df.sort_values(['subject_id','dischtime'])
df=df.reset_index(drop=True)
df.drop(['admission_type'],axis=1,inplace=True)
df=df.reset_index(drop=True)
df['readmitted']=df.groupby('subject_id').cumcount()
df=df[df['age']>=0]
df['readmitted']=df['readmitted'].shift(-1)
df['readmitted']=df['readmitted'].fillna(0)
df['readmitted']=df['readmitted'].astype(int)
df=df.reset_index(drop=True)
df=df[['readmitted', 'subject_id', 'emergency','hadm_id', 'insurance',
       'timeofstay', 'age', 'diagnosis', 'admittime', 'dischtime'
       ]]
df = df.loc[:,~df.columns.duplicated()]
def calc_read_within_6(test):
    test=test.reset_index(drop=True)
    test['total_em_6']=np.zeros(test.shape[0])
    for x in range(1,test.shape[0]):
        for j in range(0,x):
            a=(test.loc[x,'admittime']-test.loc[j,'dischtime']).days
            if a<365 and test.loc[j,'emergency']==1:
                test.loc[x,'total_em_6']=test.loc[x,'total_em_6']+1
    return test
gp = df.groupby('subject_id')   
df1=pd.DataFrame(columns=['subject_id','emergency','admittime','dischtime'])
reoccuring_ids=list(df['subject_id'].value_counts()[df['subject_id'].value_counts()>1].index)
for x in reoccuring_ids:
    a=gp.get_group(x)[['subject_id','emergency','admittime','dischtime']]
    df1=pd.concat([df1,calc_read_within_6(a)],axis=0)

df1=df1.sort_values(by='subject_id')[['subject_id','admittime','total_em_6']]
df=pd.merge(df,df1,on=['subject_id','admittime'],how='left')
df['total_em_6'].fillna(0,inplace=True)
comorbid=list(drgcodes[drgcodes['description'].str.contains(' CC| MCC')]['subject_id'])
comorbid.sort()
for x in range(0,df.shape[0]):
    if df.loc[x,'subject_id'] in comorbid:
        df.loc[x,'comorbid']=1
    else:
        df.loc[x,'comorbid']=0
diagnoses=pd.read_csv('C:/Users/utsavd/Downloads/diagnoses.csv')

diagnoses.drop(['row_id','seq_num'],axis=1,inplace=True)

diagnoses=diagnoses.sort_values(by='subject_id')
df2=pd.merge(df,diagnoses,on=['subject_id','hadm_id'],how='left')
#Charlson's Comorbidity Points Converter
def points_tally(x):
    if x.startswith(('2500','428','410','490','491','492','493','494','495','496','4439','436','43310','290','294','331','710','533')):
        return 1
    elif x.startswith(('2504','2505','2506','2507','2509','342','5853','5854','5855','5856','571','202','204','205','206','207','208')):
        return 2
    elif x.startswith('042'):
        return 5
    elif x.startswith(('190','191','192','193','194','195','196','197','198','199')):
        return 6
    else:
        return 0
df2['icd9_code']=df2['icd9_code'].astype(str)
df2['icd9_code']=df2['icd9_code'].apply(points_tally)
df=pd.merge(df,df2.groupby(['subject_id','hadm_id'])['icd9_code'].sum().reset_index(),on=['subject_id','hadm_id'],how='left')
#df.drop(['diagnosis'],axis=1,inplace=True)
for x in range(0,df.shape[0]):
    if df.loc[x,'readmitted']>0:
        df.loc[x,'t_read']=df.loc[x+1,'admittime']-df.loc[x,'dischtime']        
df['t_read']=df['t_read'].dt.days
def classify_for_readmits_within_30(x):
    if x['readmitted']>0 and x['t_read']<30.0:
        return 1
    else:
        return 0
df.readmitted=df.apply(classify_for_readmits_within_30,axis=1)
df['CHRONIC']=np.where(df['diagnosis'].str.contains('CHRONIC',na=False),1,0)
df.fillna(0,inplace=True)
icu=d['icuStays'][['subject_id','outtime']]
icu['outtime']=pd.to_datetime(icu['outtime'])
def insurance_class(x):
    if x == 'Medicare' or x == 'Medicaid':
        return 1
    else:
        return 0
df['insurance']=df['insurance'].apply(insurance_class)
import seaborn as sns
import matplotlib.pyplot as plt
input_data=df.drop(['subject_id','hadm_id','admittime','dischtime','diagnosis','t_read'],axis=1)
input_data.total_em_6=input_data['total_em_6'].astype(int)
input_data.comorbid=input_data.comorbid.astype(int)
import collections
from sklearn.model_selection import train_test_split
x=input_data.drop(['readmitted'],axis=1)
y=input_data['readmitted']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x=scaler.fit_transform(x)
X_train, X_test, Y_train, Y_test = train_test_split(x,y , test_size=0.2)
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='minority')
X_sm, Y_sm = smote.fit_sample(X_train,Y_train)
X_final, Y_final = smote.fit_sample(X_sm,Y_sm)
X_train=X_final
Y_train=Y_final
from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression(penalty='l1',max_iter=1000)
LogReg.fit(X_train, Y_train)
Y_pred = LogReg.predict(X_test)
print('Accuracy:',LogReg.score(X_test,Y_test).round(2)*100,'%')
list(zip(LogReg.coef_.ravel().round(2).tolist(),list(input_data.drop('readmitted',axis=1))))
import pickle
pickle.dump(LogReg, open('interface.sav', 'wb'))
sns.set()
plt.figure(figsize=(15,10))
ax=sns.barplot(y=l,x=['No Readmission','Readmission Within One Month'])
plt.ylim(0,1)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
    p.get_height()+0.02,
    '{:1.1f}%'.format(height*100),
    ha="center",color='black',fontsize=30)
a=list(LogReg.coef_.round(2).ravel())
b=list(X_test[3].round(2))

factors=[0,0,0,0,0,0,0,0]
for x in range(0,len(list(LogReg.coef_.round(2).ravel()))):
    factors[x]=a[x]*b[x]

s = sum(factors); norm = [float(i)/s for i in factors]

sns.set()
plt.figure(figsize=(15,10))
ax=sns.barplot(y=norm,x=list(input_data.drop('readmitted',axis=1)))
plt.ylim(0,1)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
    p.get_height()+0.02,
    '{:1.1f}%'.format(height*100),
    ha="center",color='black',fontsize=30)
pd.crosstab(Y_pred,Y_test,normalize='columns').round(2)
pd.crosstab(Y_pred,Y_test,normalize='index').round(2)
from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test, Y_pred)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()

mlp.fit(X_train,Y_train)
Y_pred = mlp.predict(X_test)
mlp.score(X_test,Y_test)
pd.crosstab(Y_pred,Y_test)
from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier(alpha=0.1,class_weight='balanced')
sgd.fit(X_train,Y_train)
Y_pred=sgd.predict(X_test)
pd.crosstab(Y_pred,Y_test,normalize='columns').round(2)
sgd.score(X_test,Y_test)
from sklearn.metrics import classification_report 
print(classification_report(Y_pred,Y_test))
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model = XGBClassifier()
model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)
accuracy_score(Y_pred,Y_test)
from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test, Y_pred)
plt.figure(figsize=(15,10))
df.groupby('age')['readmitted'].mean().plot.bar()
pd.crosstab(Y_pred,Y_test)
from sklearn.neighbors import KNeighborsClassifier
## Instantiate the model with 5 neighbors. 
knn = KNeighborsClassifier(weights='distance')
## Fit the model on the training data.
knn.fit(X_train, Y_train)
## See how the model performs on the test data.
knn.score(X_test, Y_test)
Y_pred=knn.predict(X_test)
from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test, Y_pred)
from sklearn.svm import SVC
svc = SVC(class_weight='balanced')
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_test,Y_test), 3)
acc_svc
from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test, Y_pred)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=28, criterion = "entropy", min_samples_split=10)
tree.fit(X_train,Y_train)
tree.score(X_test,Y_test)
roc_auc_score(Y_pred,Y_test)
pd.crosstab(Y_test,Y_pred)
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(class_weight='balanced')
forest.fit(X_train,Y_train)
forest.score(X_test,Y_test)
Y_pred=forest.predict(X_test)
pd.crosstab(Y_pred,Y_test,normalize='columns')
