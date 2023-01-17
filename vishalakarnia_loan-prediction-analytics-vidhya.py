import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

l_train=pd.read_csv('../input/train_ctrUa4K.csv')

l_train.head()
l_train.isnull().sum()
l_train.dtypes
l_train.Dependents.unique()
sns.heatmap(l_train.isnull())

plt.figure(figsize=(20,10))
l_train['Gender'].value_counts()
l_train.Gender.unique()
# l_train['Gender'].replace(np.nan,'Male',inplace=True)
sns.barplot(hue='Gender',x='Self_Employed',y='CoapplicantIncome',data=l_train)
sns.barplot(hue='Gender',x='Education',y='CoapplicantIncome',data=l_train)
l_train['Gender'].replace(np.nan,'Male',inplace=True)
l_train['Married'].value_counts()
sns.barplot(x='Married',y='CoapplicantIncome',data=l_train)
l_train.loc[l_train['Married'].isnull()]
l_train.loc[104,'Married']='Yes'

l_train['Married'].replace(np.nan,'No',inplace=True)
l_train.isnull().sum()
sns.barplot(x='Dependents',y='CoapplicantIncome',hue='Married',data=l_train)
l_train['Dependents'].value_counts()
l_train.loc[l_train['Dependents'].isnull()]
l_train.loc[228,'Dependents']='3+'

l_train.loc[293,'Dependents']='3+'

l_train.loc[332,'Dependents']='3+'

l_train.loc[355,'Dependents']='3+'

l_train.loc[435,'Dependents']='3+'

l_train['Dependents'].replace(np.nan,'0',inplace=True)
sns.heatmap(l_train.isnull())

plt.figure(figsize=(20,10))
l_train[l_train['Self_Employed'].isnull()]

l_train['Self_Employed'].replace(np.nan,'DK',inplace=True)

# df

# df['Self_Employed']=['Yes' if i =='Graduate' else 'No' for i in df['Education']]

# df.head()
sns.barplot(x='Self_Employed',y='ApplicantIncome',hue= 'Education',data=l_train)
sns.barplot(hue='Self_Employed',x='Education',y='LoanAmount',data=l_train)
l_train.loc[(l_train.Self_Employed == "DK") & (l_train.Education == "Not Graduate")]



# l_train.loc[(l_train.Self_Employed == "DK") & (l_train.Education == "Not Graduate")].replace('DK','No',inplace=True)
# l_train.loc[(l_train.Self_Employed == "DK") & (l_train.Education == "Graduate")]





# l_train.loc[(l_train.Self_Employed == "DK") & (l_train.Education == "Graduate")].replace('DK','Yes',inplace=True)
# Self_Employed_=[]

# for i in l_train.Self_Employed:

#     for j in l_train.Education:

#         if i == 'Dk' and j == 'Not Graduate':

#             Self_Employed_.append("No")

#         elif i== 'DK' and j =='Graduate':

#             Self_Employed_.append("Yes")

#         else:

#             Self_Employed_.append(i)
l_train.loc[107,'Self_Employed']='No'

l_train.loc[170,'Self_Employed']='No'

l_train.loc[463,'Self_Employed']='No'

l_train.loc[468,'Self_Employed']='No'

l_train.loc[535,'Self_Employed']='No'

l_train.loc[601,'Self_Employed']='No'
l_train['Self_Employed'].replace('DK','Yes',inplace=True)
l_train['Self_Employed'].value_counts()
sns.barplot(x='Self_Employed',y='ApplicantIncome',hue= 'Education',data=l_train)
l_train['Self_Employed'].isnull().sum()
l_train.isnull().sum()
sns.heatmap(l_train.isnull())

plt.figure(figsize=(20,10))
l_train['ApplicantIncome'].mean()
l_train['LoanAmount'].replace(np.nan,l_train['LoanAmount'].median(),inplace=True)
l_train.isnull().sum()
l_train['Loan_Amount_Term'].value_counts()
sns.barplot(hue='Self_Employed',y='Loan_Amount_Term',x='Married',data=l_train)
l_train['Loan_Amount_Term'].fillna(l_train['Loan_Amount_Term'].median(),inplace=True)
l_train.isnull().sum()
# l_train['Credit_History']=l_train['Credit_History'].astype(str,inplace=True)
l_train.dtypes
l_train['Credit_History'].isnull().sum()
sns.barplot(hue='Self_Employed',y='CoapplicantIncome',x='Credit_History',data=l_train)
l_train['Credit_History'].replace('nan',np.nan,inplace=True)
l_train.isnull().sum()
sns.heatmap(l_train.isnull())

plt.figure(figsize=(100,100))
l_train.loc[l_train['Credit_History'].isnull()]
l_train['Loan_Status'].replace('Y',1,inplace=True)

l_train['Loan_Status'].replace('N',0,inplace=True)
l_train['Credit_History'].fillna(l_train['Credit_History'].mode()[0],inplace=True)
sns.heatmap(l_train.isnull())

plt.figure(figsize=(100,100))
l_train['Loan_Status']=l_train['Loan_Status'].astype(int,inplace=True)
l_train.dtypes
# Gender=pd.get_dummies(l_train.Gender,prefix="Gender").iloc[ : ,1:]

# Married=pd.get_dummies(l_train.Married,prefix="married").iloc[ : ,1:]

# Dependents=pd.get_dummies(l_train.Dependents,prefix="Dependents").iloc[ : ,1:]

# Education=pd.get_dummies(l_train.Education,prefix="Education").iloc[ : ,1:]

# Self_Employed=pd.get_dummies(l_train.Self_Employed,prefix="Self_Employed").iloc[ : ,1:]

# Property_Area=pd.get_dummies(l_train.Property_Area,prefix="Property_Area").iloc[ : ,1:]

# l_train=pd.concat([l_train,Gender],axis=1)

# l_train=pd.concat([l_train,Married],axis=1)

# l_train=pd.concat([l_train,Dependents],axis=1)

# l_train=pd.concat([l_train,Education],axis=1)

# l_train=pd.concat([l_train,Self_Employed],axis=1)

# l_train=pd.concat([l_train,Property_Area],axis=1)

# l_train.head()
l_train=pd.get_dummies(l_train,columns=["Gender","Married","Dependents","Education","Self_Employed","Property_Area"],drop_first=True)

l_train.head()
l_train.corr()
pd.crosstab(l_train['Credit_History'],l_train['Loan_Status'],margins=True)
x=l_train.drop(['Loan_Status','Loan_ID','ApplicantIncome','CoapplicantIncome','LoanAmount',

               'Loan_Amount_Term','Dependents_1','Dependents_3+','Education_Not Graduate',

               'Self_Employed_Yes','Property_Area_Urban'],axis=1)

y=l_train['Loan_Status']
from sklearn.model_selection import train_test_split





x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 21)
from sklearn.neighbors import KNeighborsClassifier



clf = KNeighborsClassifier(n_neighbors= 7)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
print('Accuracy_score: ',accuracy_score(y_test, y_pred)*100)

print('\n','Classification_report: ','\n','\n',classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

print(cm) 
l_test=pd.read_csv('../input/test_lAUu6dG.csv')

l_test.head()
l_test.isnull().sum()
l_test.dtypes
l_test['Gender'].unique()
l_test['Gender'].value_counts()
sns.barplot(hue='Gender',x="Self_Employed",y="ApplicantIncome",data=l_test)
l_test['Gender'].replace(np.nan,'Male',inplace=True)
l_test['Gender'].unique()
l_test['Dependents'].unique()
sns.barplot(x='Dependents',y='ApplicantIncome',hue='Married',data=l_test)
l_test.loc[l_test['Dependents'].isnull()]

l_test.loc[46,'Dependents']='3+'

l_test.loc[111,'Dependents']='3+'

l_test.loc[202,'Dependents']='3+'

l_test.loc[247,'Dependents']='3+'

l_test.loc[251,'Dependents']='3+'
l_test['Dependents'].replace(np.nan,'0',inplace=True)



l_test.isnull().sum()
l_test['Self_Employed'].replace(np.nan,'dk',inplace=True)

l_test.loc[l_test['Self_Employed']=='dk']
sns.barplot(x='Self_Employed',y='ApplicantIncome',hue='Education',data=l_test)
sns.barplot(x='Self_Employed',y='ApplicantIncome',hue='Married',data=l_test)
sns.barplot(x='Property_Area',y='ApplicantIncome',hue='Self_Employed',data=l_test)
sns.barplot(x='Self_Employed',y='ApplicantIncome',hue='Gender',data=l_test)
l_test.loc[89,'Self_Employed']='No'

l_test.loc[168,'Self_Employed']='No'

l_test.loc[259,'Self_Employed']= 'No'



l_test['Self_Employed'].replace('dk','Yes',inplace=True)
l_test.isnull().sum()
l_test['LoanAmount'].replace(np.nan,l_test['LoanAmount'].median(),inplace=True)
l_test['Loan_Amount_Term'].replace(np.nan,l_test['Loan_Amount_Term'].median(),inplace=True)
l_test.isnull().sum()
l_test[l_test['Credit_History'].isnull()]
l_test['Credit_History'].value_counts()
l_test['Credit_History'].replace(np.nan,1.0,inplace=True)
l_test.isnull().sum()
l_test.head()
l_test.dtypes
l_test=pd.get_dummies(l_test,columns=["Gender","Married","Dependents","Education","Self_Employed","Property_Area"],drop_first=True)

l_test.head()
df_test=pd.get_dummies(l_test,drop_first=True)

df_test=l_test.drop(['Loan_ID','ApplicantIncome','CoapplicantIncome','LoanAmount',

               'Loan_Amount_Term','Dependents_1','Dependents_3+','Education_Not Graduate',

               'Self_Employed_Yes','Property_Area_Urban'],axis=1)
df_test.head()
y_pred_test = clf.predict(df_test)
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
Y=pd.DataFrame(y_pred_test)
Y.head(10)
sm=pd.read_csv("../input/sample_submission_49d68Cx.csv")
sm.head()
submit=pd.concat([sm.Loan_ID,Y],axis=1)
submit.head()
submit.columns=["Loan_ID","Loan_Status"]
submit.head()

submit['Loan_Status'].replace(0,'N',inplace=True)



submit['Loan_Status'].replace(1,'Y',inplace=True)
submit.head()
submit['Loan_Status'].value_counts()
submit.to_csv("Submission1.csv",index=False)
import pandas as pd

sample_submission_49d68Cx = pd.read_csv("../input/sample_submission_49d68Cx.csv")

test_lAUu6dG = pd.read_csv("../input/test_lAUu6dG.csv")

train_ctrUa4K = pd.read_csv("../input/train_ctrUa4K.csv")