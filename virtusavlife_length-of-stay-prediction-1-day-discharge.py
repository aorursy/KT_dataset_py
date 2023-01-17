import requests
import pandas as pd
import json
import matplotlib as plt
import seaborn as sns
from pandas.io.json import json_normalize
url1 = "http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/admissions?limit=50000&offset=0"
conn1 = requests.get(url1).json()
file1 = conn1['admissions']
admissions =  pd.DataFrame.from_dict(file1, orient='columns')
url = "http://ec2-54-88-151-77.compute-1.amazonaws.com:3001/v1/admissions?limit=50000&offset=0"

url1 = "http://ec2-54-88-151-77.compute-1.amazonaws.com:3003/v1/eligibilities?limit=50000&offset=0"

url2 = "http://ec2-54-88-151-77.compute-1.amazonaws.com:3003/v1/conditions?limit=50000&offset=0"
#conn = requests.get(url).json()
#conn1 = requests.get(url1).json()
#conn2 = requests.get(url2).json()
file = conn1['admissions']
file1 = conn1['eligibility']
file2 = conn1['conditions']
admissions =  pd.DataFrame.from_dict(file, orient='columns')
eligibilities =  pd.DataFrame.from_dict(file1, orient='columns')
conditions =  pd.DataFrame.from_dict(file2, orient='columns')
admissions = pd.read_table(r'C:\Users\venkataavinashy\Desktop\Vlife\MIMIC-III_readmission-master\MIMIC-III_readmission-master\notebooks\Tables\admissions.csv', delimiter = ',')
admissions.head(3)
admissions.discharge_location.value_counts()
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['los'] = admissions['dischtime'] - admissions['admittime']
onedayadmits = admissions[admissions.los.astype('timedelta64[D]')<1 & admissions.los.notnull()]
onedayadmits['los'] = onedayadmits['los'].apply(lambda x: x.total_seconds())
onedayadmits['los'] = onedayadmits['los']/3600
onedayadmits = onedayadmits[onedayadmits.los > 1]

#Dropping columns
onedayadmits = onedayadmits.drop(['row_id','edregtime','edouttime','dischtime','deathtime'],axis=1)
onedayadmits.head(1)
%matplotlib inline
onedayadmits['los'].hist(bins = 100)
onedayadmits.hospital_expire_flag.value_counts()
diedinhosp  = onedayadmits[onedayadmits.hospital_expire_flag == 1 ]
diedinhosp.los.hist(bins=50)
onedayadmits.diagnosis = onedayadmits.diagnosis.fillna('')
onedayadmits.religion.value_counts()
onedayadmits.religion = onedayadmits.religion.fillna('OTHER')
sns.set(style="ticks", palette="pastel")


g = sns.barplot(onedayadmits.religion.value_counts().index, onedayadmits['religion'].value_counts().values)
g.set_xticklabels(labels = onedayadmits.religion.value_counts().index.tolist(), rotation=90)
import seaborn as sns
sns.set(style="ticks", palette="pastel")
sns.boxplot
g = sns.boxplot(x="religion", y="los", data=onedayadmits)
sns.despine(offset=10, trim=True)
g.set_xticklabels(labels = onedayadmits.religion.value_counts().index.tolist(), rotation=90)
g = sns.boxplot(x="ethnicity", y="los", data=onedayadmits)
sns.despine(offset=10, trim=True)
g.set_xticklabels(labels = onedayadmits.ethnicity.value_counts().index.tolist(), rotation=90)
onedayadmits.language = onedayadmits.language.fillna('NONE')
onedayadmits.language.value_counts()
g = sns.boxplot(x="language", y="los", data=onedayadmits)
sns.despine(offset=10, trim=True)
g.set_xticklabels(labels = onedayadmits.language.value_counts().index.tolist(), rotation=90)
onedayadmits.marital_status = onedayadmits.marital_status.fillna('NONE')
g = sns.boxplot(x="marital_status", y="los", data=onedayadmits)
sns.despine(offset=10, trim=True)
g.set_xticklabels(labels = onedayadmits.marital_status.value_counts().index.tolist(), rotation=90)
onedayadmits.info()
g = sns.boxplot(x="hospital_expire_flag", y="los", data=onedayadmits)
sns.despine(offset=10, trim=True)
g.set_xticklabels(labels = onedayadmits.hospital_expire_flag.value_counts().index.tolist(), rotation=90)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
onedayadmits.marital_status = label_encoder.fit_transform(onedayadmits.marital_status)
onedayadmits.admission_location = label_encoder.fit_transform(onedayadmits.admission_location)
onedayadmits.admission_type = label_encoder.fit_transform(onedayadmits.admission_type)
onedayadmits.diagnosis= label_encoder.fit_transform(onedayadmits.diagnosis)
onedayadmits.discharge_location= label_encoder.fit_transform(onedayadmits.discharge_location)
onedayadmits.ethnicity= label_encoder.fit_transform(onedayadmits.ethnicity)
onedayadmits.insurance= label_encoder.fit_transform(onedayadmits.insurance)
onedayadmits.language= label_encoder.fit_transform(onedayadmits.language)
onedayadmits.religion= label_encoder.fit_transform(onedayadmits.religion)
onedayadmits.admittime= label_encoder.fit_transform(onedayadmits.admittime)
train = pd.DataFrame(onedayadmits, columns=['admission_location','admission_type','admittime','diagnosis','discharge_location','ethnicity','hospital_expire_flag','insurance','language','marital_status','religion'])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
X_train, X_test, y_train, y_test = train_test_split(train, onedayadmits['los'], test_size=0.2, random_state=42)

onedayadmits.info()
regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
# The coefficients
print('Coefficients: \n', regr.coef_)
print('\n \nScore: \n', regr.score(X_test,y_test))


# Plot outputs
#plt.scatter(X_test, y_test,  color='black')
#plt.plot(X_test, y_pred, color='blue', linewidth=3)

#plt.xticks(())
#plt.yticks(())

#plt.show()
sns.distplot(y_pred,color="y")
