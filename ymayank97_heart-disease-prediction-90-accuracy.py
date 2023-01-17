#importing libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import export_graphviz #plot tree

from sklearn.metrics import roc_curve, auc #for model evaluation

from sklearn.metrics import classification_report #for model evaluation

from sklearn.metrics import confusion_matrix #for model evaluation

from sklearn.model_selection import train_test_split #for data splitting

from sklearn.metrics import r2_score,accuracy_score

from sklearn.model_selection import cross_val_score ,StratifiedKFold

from scipy import stats

import pylab

import eli5 #for permutation importance

from eli5.sklearn import PermutationImportance

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

#reading dataset

dataset=pd.read_csv("../input/heart.csv")

dataset.columns=['Age', 'Gender', 'CP', 'Trestbps', 'Chol', 'FBS', 'RestECG',

                 'Thalach', 'Exang', 'Oldpeak', 'Slope', 'CA', 'Thal', 'Goal']

nRow, nCol = dataset.shape



n_with_disease = dataset[dataset["Goal"]==2].shape[0]

n_without_disease = dataset[dataset["Goal"]==1].shape[0]

greater_percent = (n_with_disease*100)/float(nRow)



print(f'**Summary**:\n There are {nRow} rows and {nCol} columns. Goal is the target/label variable that can have only value(1/2)')

disease = len(dataset[dataset['Goal'] == 1])

non_disease = len(dataset[dataset['Goal'] == 0])

plt.pie(x=[disease, non_disease], explode=(0, 0), labels=['Diseased ', 'Non-diseased'], autopct='%1.2f%%', shadow=True, startangle=90)

plt.show()
#Check sample of any 5 rows

dataset=dataset.reset_index()

dataset=dataset.drop(['index'],axis=1)

dataset.sample(5)
# Get the number of missing data points, NA's ,NAN's values per column

total = dataset.isnull().sum().sort_values(ascending=False)

percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



total = dataset.isna().sum().sort_values(ascending=False)

percent = (dataset.isna().sum()/dataset.isna().count()).sort_values(ascending=False)

na_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



if((na_data.all()).all()>0 or (na_data.all()).all()>0):

     print('Found Missing Data or NA values')

#print(na_data,"\n",missing_data)
#Detect outliers

plt.subplots(figsize=(18,10))

dataset.boxplot(patch_artist=True, sym="k.")

plt.xticks(rotation=90)

df=dataset[~(np.abs(stats.zscore(dataset)) < 3).all(axis=1)]

df
dataset=dataset.drop(dataset[~(np.abs(stats.zscore(dataset)) < 3).all(axis=1)].index)
dataset['Gender']=dataset['Gender'].replace([1,0], ['Male', 'Female'])

dataset['Goal']=dataset['Goal'].replace([0,1], ['Absence', 'Presence'])

dataset['Slope']=dataset['Slope'].replace([0,1,2], ['Upsloping','Flat','Down-sloping'])

dataset['RestECG']=dataset['RestECG'].replace([0,1,2], ['Normal', 'Abnormality','Hypertrophy'])

dataset['Exang']=dataset['Exang'].replace([1,0], ['Yes', 'No'])

dataset['FBS']=dataset['FBS'].replace([1,0], ['Yes', 'No'])

dataset['Thal']=dataset['Thal'].replace([1,2,3], ['Normal', 'Fixed Defect','Reversible defect'])

dataset['CP']=dataset['CP'].replace([0,1,2,3], ['Typical angina', 'Atypical angina','Non-anginal pain','Asymptomatic pain'])



dataset['Gender']=dataset['Gender'].astype('object')

dataset['CP']=dataset['CP'].astype('object')

dataset['Thal']=dataset['Thal'].astype('object')

dataset['FBS']=dataset['FBS'].astype('object')

dataset['Exang']=dataset['Exang'].astype('object')

dataset['RestECG']=dataset['RestECG'].astype('object')

dataset['Slope']=dataset['Slope'].astype('object')
cont_dataset=dataset.copy()

cont_dataset=cont_dataset.drop(['Gender','Slope','Thal','CP','FBS','RestECG','Exang','Goal'],axis=1)

plt.subplots(figsize=(10,8))

colr=sns.heatmap(cont_dataset.corr(),robust=True,annot=True)

figure = colr.get_figure()    

figure.savefig('correlation.png', dpi=400)
pd.crosstab(dataset['Age'],dataset['Goal']).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

# plt.savefig('heartDiseaseAndAges.png')

plt.show()
absence = dataset[dataset['Goal']=='Absence']['Age']

presence = dataset[dataset['Goal']=='Presence']['Age']



fig, ax = plt.subplots(1,2,figsize=(16,6))

mean=round(dataset['Age'].mean(),2)

median=dataset['Age'].median()

a_median=absence.median()

a_mean=round(absence.mean(),2)

p_median=presence.median()

p_mean=round(presence.mean(),2)



ax[1].axvline(p_median, color='r', linestyle='-')

ax[1].axvline(p_mean, color='r', linestyle='--')

ax[1].axvline(a_median, color='b', linestyle='-')

ax[1].axvline(a_mean, color='b', linestyle='--')

ax[0].axvline(median, color='g', linestyle='-')

ax[0].axvline(mean, color='g', linestyle='--')



ax[0]=sns.distplot(dataset['Age'],bins=15,ax=ax[0])

ax[1]=sns.kdeplot(absence, label='Absence', shade=True)

ax[1]=sns.kdeplot(presence, label='Presence',shade=True)



plt.xlabel('Age');

plt.show()

fig.savefig('age_old.png')

print(f' \tMean & Median of whole dataset are {mean} & {median}\t\t\tMean & Median of absence data = {a_mean} & {a_median}\n\t\t\t\t\t\t\t\t\tMean & Median of presence data are {p_mean} & {p_median}  ')
print(f"Normal Test for the Age distribution {stats.normaltest(dataset['Age'])}") #Null hypothesis : data came from a normal distribution.



print(f"Skewness for the whole dataset {pd.DataFrame.skew(dataset['Age'], axis=0)}") #left skewed

print(f"Skewness of non-disease cohort {pd.DataFrame.skew(absence, axis=0)}")  #right skewed

print(f"Skewness of disease cohort{pd.DataFrame.skew(presence, axis=0)}")   #left skewed



#If the skewness is between -0.5 and 0.5, the data are fairly symmetrical
absenceMeans = []

presenceMeans = []

sampleMeans=[]

for _ in range(1000):

    samples = dataset['Age'].sample(n=200)

    sampleMean = np.mean(samples)

    sampleMeans.append(sampleMean)

    

    samples = absence.sample(n=100)

    sampleMean = np.mean(samples)

    absenceMeans.append(sampleMean)

    

    samples = presence.sample(n=100)

    sampleMean = np.mean(samples)

    presenceMeans.append(sampleMean)





    

fig, ax = plt.subplots(1,2,figsize=(16,6))

ax[1].axvline(p_median, color='r', linestyle='-')

ax[1].axvline(p_mean, color='r', linestyle='--')

ax[1].axvline(a_median, color='b', linestyle='-')

ax[1].axvline(a_mean, color='b', linestyle='--')



ax[0].axvline(p_median, color='r', linestyle='-')

ax[0].axvline(p_mean, color='r', linestyle='--')

ax[0].axvline(a_median, color='b', linestyle='-')

ax[0].axvline(a_mean, color='b', linestyle='--')



ax[0] =sns.kdeplot(absence, label='Absence', shade=True,ax=ax[0])

ax[0] =sns.kdeplot(presence, label='Presence',shade=True,ax=ax[0])



ax[1] =sns.kdeplot(absenceMeans, label='Absence', shade=True,ax=ax[1])

ax[1] =sns.kdeplot(presenceMeans, label='Presence',shade=True,ax=ax[1])

ax[0].set_xlabel('Age')

ax[0].set_ylabel('Kernel Density Estimate')

ax[1].set_xlabel('Age')

ax[1].set_ylabel('Kernel Density Estimate')

ax[0].set_title('Before Central Limit Theorem')

ax[1].set_title('After Central Limit Theorem')

plt.show()

fig.savefig('age.png')

print(f' \tMean & Median of whole dataset are {mean} & {median}\t\t\tMean & Median of absence data = {a_mean} & {a_median}\n\t\t\t\t\t\t\t\t\tMean & Median of presence data are {p_mean} & {p_median}  ')
#t-test on independent samples

t2, p2 = stats.ttest_ind(presence,absence)

print("t = " + str(t2))

print("p = " + str(2*p2))
print(f"Normal Test for the whole dataset {stats.normaltest(dataset['Trestbps'])}") #Null hypothesis : data came from a normal distribution.
absence = dataset[dataset['Goal']=='Absence']['Trestbps']

presence = dataset[dataset['Goal']=='Presence']['Trestbps']



a_median=absence.median()

a_mean=round(absence.mean(),2)

p_median=presence.median()

p_mean=round(presence.mean(),2)

absenceMeans = []

presenceMeans = []

sampleMeans=[]

for _ in range(1000):

    samples = dataset['Trestbps'].sample(n=100)

    sampleMean = np.mean(samples)

    sampleMeans.append(sampleMean)

    

    samples = absence.sample(n=100)

    sampleMean = np.mean(samples)

    absenceMeans.append(sampleMean)

    

    samples = presence.sample(n=100)

    sampleMean = np.mean(samples)

    presenceMeans.append(sampleMean)

    

fig, ax = plt.subplots(1,2,figsize=(16,6))



ax[1].axvline(p_median, color='r', linestyle='-')

ax[1].axvline(p_mean, color='r', linestyle='--')

ax[1].axvline(a_median, color='b', linestyle='-')

ax[1].axvline(a_mean, color='b', linestyle='--')



ax[0].axvline(p_median, color='r', linestyle='-')

ax[0].axvline(p_mean, color='r', linestyle='--')

ax[0].axvline(a_median, color='b', linestyle='-')

ax[0].axvline(a_mean, color='b', linestyle='--')



ax[0] =sns.kdeplot(absence, label='Absence', shade=True,ax=ax[0])

ax[0] =sns.kdeplot(presence, label='Presence',shade=True,ax=ax[0])



ax[1] =sns.kdeplot(absenceMeans, label='Absence', shade=True)

ax[1] =sns.kdeplot(presenceMeans, label='Presence',shade=True)

ax[0].set_xlabel('Trestbps')

ax[0].set_ylabel('Kernel Density Estimate')

ax[1].set_xlabel('Trestbps')

ax[1].set_ylabel('Kernel Density Estimate')

ax[0].set_title('Before Central Limit Theorem')

ax[1].set_title('After Central Limit Theorem')

plt.show()

fig.savefig('Trestbps.png')

print(f' \tMean & Median of absence data = {a_mean} & {a_median}\t\t\tMean & Median of presence data are {p_mean} & {p_median}  ')
#t-test on independent samples

t2, p2 = stats.ttest_ind(presenceMeans,absenceMeans)

print("t = " + str(t2))

print("p = " + str(2*p2))
print(f"Normal Test for the whole dataset {stats.normaltest(dataset['Chol'])}") #Null hypothesis : data came from a normal distribution.
absence = dataset[dataset['Goal']=='Absence']['Chol']

presence = dataset[dataset['Goal']=='Presence']['Chol']



a_median=absence.median()

a_mean=round(absence.mean(),2)

p_median=presence.median()

p_mean=round(presence.mean(),2)

absenceMeans = []

presenceMeans = []

sampleMeans=[]

for _ in range(1000):

    samples = dataset['Chol'].sample(n=100)

    sampleMean = np.mean(samples)

    sampleMeans.append(sampleMean)

    

    samples = absence.sample(n=100)

    sampleMean = np.mean(samples)

    absenceMeans.append(sampleMean)

    

    samples = presence.sample(n=100)

    sampleMean = np.mean(samples)

    presenceMeans.append(sampleMean)

    

fig, ax = plt.subplots(1,2,figsize=(16,6))



ax[1].axvline(p_median, color='r', linestyle='-')

ax[1].axvline(p_mean, color='r', linestyle='--')

ax[1].axvline(a_median, color='b', linestyle='-')

ax[1].axvline(a_mean, color='b', linestyle='--')



ax[0].axvline(p_median, color='r', linestyle='-')

ax[0].axvline(p_mean, color='r', linestyle='--')

ax[0].axvline(a_median, color='b', linestyle='-')

ax[0].axvline(a_mean, color='b', linestyle='--')



ax[0] =sns.kdeplot(absence, label='Absence', shade=True,ax=ax[0])

ax[0] =sns.kdeplot(presence, label='Presence',shade=True,ax=ax[0])



ax[1] =sns.kdeplot(absenceMeans, label='Absence', shade=True)

ax[1] =sns.kdeplot(presenceMeans, label='Presence',shade=True)

ax[0].set_xlabel('Chol')

ax[0].set_ylabel('Kernel Density Estimate')

ax[1].set_xlabel('Chol')

ax[1].set_ylabel('Kernel Density Estimate')

ax[0].set_title('Before Central Limit Theorem')

ax[1].set_title('After Central Limit Theorem')

plt.show()

fig.savefig('chol.png')

print(f' \tMean & Median of absence data = {a_mean} & {a_median}\t\t\tMean & Median of presence data are {p_mean} & {p_median}  ')
#t-test on independent samples

t2, p2 = stats.ttest_ind(presenceMeans,absenceMeans)

print("t = " + str(t2))

print("p = " + str(2*p2))
pd.crosstab(dataset['Oldpeak'],dataset['Goal']).plot(kind="bar",figsize=(10,6))

plt.title('Heart Disease Frequency for Oldpeak')

plt.xlabel('Exercise Induced ST Depression')

plt.ylabel('Frequency')

plt.show()
absence = dataset[dataset['Goal']=='Absence']['Oldpeak']

presence = dataset[dataset['Goal']=='Presence']['Oldpeak']



a_median=absence.median()

a_mean=round(absence.mean(),2)

p_median=presence.median()

p_mean=round(presence.mean(),2)

absenceMeans = []

presenceMeans = []

sampleMeans=[]

for _ in range(1000):

    samples = dataset['Oldpeak'].sample(n=100)

    sampleMean = np.mean(samples)

    sampleMeans.append(sampleMean)

    

    samples = absence.sample(n=100)

    sampleMean = np.mean(samples)

    absenceMeans.append(sampleMean)

    

    samples = presence.sample(n=100)

    sampleMean = np.mean(samples)

    presenceMeans.append(sampleMean)

    

fig, ax = plt.subplots(1,2,figsize=(16,6))



ax[1].axvline(p_median, color='r', linestyle='-')

ax[1].axvline(p_mean, color='r', linestyle='--')

ax[1].axvline(a_median, color='b', linestyle='-')

ax[1].axvline(a_mean, color='b', linestyle='--')



ax[0].axvline(p_median, color='r', linestyle='-')

ax[0].axvline(p_mean, color='r', linestyle='--')

ax[0].axvline(a_median, color='b', linestyle='-')

ax[0].axvline(a_mean, color='b', linestyle='--')



ax[0] =sns.kdeplot(absence, label='Absence', shade=True,ax=ax[0])

ax[0] =sns.kdeplot(presence, label='Presence',shade=True,ax=ax[0])



ax[1] =sns.kdeplot(absenceMeans, label='Absence', shade=True)

ax[1] =sns.kdeplot(presenceMeans, label='Presence',shade=True)

ax[0].set_xlabel('Oldpeak')

ax[0].set_ylabel('Kernel Density Estimate')

ax[1].set_xlabel('Oldpeak')

ax[1].set_ylabel('Kernel Density Estimate')

ax[0].set_title('Before Central Limit Theorem')

ax[1].set_title('After Central Limit Theorem')

plt.show()

fig.savefig('oldpeak.png')

print(f' \tMean & Median of absence data = {a_mean} & {a_median}\t\t\tMean & Median of presence data are {p_mean} & {p_median}  ')
#t-test on independent samples

t2, p2 = stats.ttest_ind(presenceMeans,absenceMeans)

print("t = " + str(t2))

print("p = " + str(2*p2))
absence = dataset[dataset['Goal']==1]['Thalach']

absence = dataset[dataset['Goal']=='Absence']['Thalach']

presence = dataset[dataset['Goal']=='Presence']['Thalach']

mean=round(dataset['Thalach'].mean())

median=dataset['Thalach'].median()

a_median=absence.median()

a_mean=round(absence.mean(),2)

p_median=presence.median()

p_mean=round(presence.mean(),2)

fig, ax = plt.subplots(1,2,figsize=(16,6))



ax[1].axvline(p_median, color='r', linestyle='-')

ax[1].axvline(p_mean, color='r', linestyle='--')

ax[1].axvline(a_median, color='b', linestyle='-')

ax[1].axvline(a_mean, color='b', linestyle='--')

ax[0].axvline(median, color='g', linestyle='-')

ax[0].axvline(mean, color='g', linestyle='--')



ax[0]=sns.distplot(dataset['Thalach'],bins=15,ax=ax[0])

ax[1] =sns.kdeplot(absence, label='Absence', shade=True)

ax[1] =sns.kdeplot(presence, label='Presence',shade=True)

plt.show()

fig.savefig('thalach_old.png')

print(f' \tMean & Median of whole dataset are {mean} & {median}\t\t\tMean & Median of absence data = {a_mean} & {a_median}\n\t\t\t\t\t\t\t\t\tMean & Median of presence data are {p_mean} & {p_median}  ')

# Oldpeak: ST depression induced by exercise relative to rest
absenceMeans = []

presenceMeans = []

sampleMeans=[]

for _ in range(1000):

    samples = dataset['Thalach'].sample(n=100)

    sampleMean = np.mean(samples)

    sampleMeans.append(sampleMean)

    

    samples = absence.sample(n=100)

    sampleMean = np.mean(samples)

    absenceMeans.append(sampleMean)

    

    samples = presence.sample(n=100)

    sampleMean = np.mean(samples)

    presenceMeans.append(sampleMean)

    

    

fig, ax = plt.subplots(1,2,figsize=(16,6))



ax[1].axvline(p_median, color='r', linestyle='-')

ax[1].axvline(p_mean, color='r', linestyle='--')

ax[1].axvline(a_median, color='b', linestyle='-')

ax[1].axvline(a_mean, color='b', linestyle='--')



ax[0].axvline(p_median, color='r', linestyle='-')

ax[0].axvline(p_mean, color='r', linestyle='--')

ax[0].axvline(a_median, color='b', linestyle='-')

ax[0].axvline(a_mean, color='b', linestyle='--')



ax[0] =sns.kdeplot(absence, label='Absence', shade=True,ax=ax[0])

ax[0] =sns.kdeplot(presence, label='Presence',shade=True,ax=ax[0])



ax[1] =sns.kdeplot(absenceMeans, label='Absence', shade=True)

ax[1] =sns.kdeplot(presenceMeans, label='Presence',shade=True)

ax[0].set_xlabel('Thalach')

ax[0].set_ylabel('Kernel Density Estimate')

ax[1].set_xlabel('Thalach')

ax[1].set_ylabel('Kernel Density Estimate')

ax[0].set_title('Before Central Limit Theorem')

ax[1].set_title('After Central Limit Theorem')

plt.show()

fig.savefig('thalach.png')

print(f' \tMean & Median of whole dataset are {mean} & {median}\t\t\tMean & Median of absence data = {a_mean} & {a_median}\n\t\t\t\t\t\t\t\t\tMean & Median of presence data are {p_mean} & {p_median}  ')
#t-test on independent samples

t2, p2 = stats.ttest_ind(presenceMeans,absenceMeans)

print("t = " + str(t2))

print("p = " + str(2*p2))
male = len(dataset[dataset['Gender'] == 'Male'])

female = len(dataset[dataset['Gender'] == 'Female'])

plt.pie(x=[male, female], explode=(0, 0), labels=['Male', 'Female'], autopct='%1.2f%%', shadow=True, startangle=90)

plt.show()
absence = dataset[dataset["Goal"]=='Absence']["Gender"].sort_values()

presence = dataset[dataset["Goal"]=='Presence']["Gender"].sort_values()

f, axes = plt.subplots(1,2,figsize=(15,5))

sns.countplot(absence, data=dataset,ax=axes[0]).set_title('Absence of Heart Disease')

sns.countplot(presence, data=dataset,ax=axes[1]).set_title('Presence of Heart Disease')

plt.show()

f.savefig('gender.png')
# Chi-square test of independence of variables

cont = pd.crosstab(dataset['Gender'],dataset['Goal'])

chi_stat = stats.chi2_contingency(cont)

print(f'Chi statistics is {chi_stat[0]} and  p value is {chi_stat[1]}')

#Null hypothesis : Gender is not associated with Goal

#Alternate hypothesis : Gender is associated with Goal 
x = [len(dataset[dataset['CP'] == 'Typical angina']),len(dataset[dataset['CP'] == 'Atypical angina']), len(dataset[dataset['CP'] == 'Non-anginal pain']), len(dataset[dataset['CP'] == 'Asymptomatic pain'])]

plt.pie(x, data=dataset, labels=['CP(1) Typical angina', 'CP(2) Atypical angina', 'CP(3) Non-anginal pain', 'CP(4) Asymptomatic pain'], autopct='%1.2f%%', shadow=True,startangle=90)

plt.show()

f, axes = plt.subplots(1,2,figsize=(15,5))

absence = dataset[dataset["Goal"]=='Absence']["CP"]

presence = dataset[dataset["Goal"]=='Presence']["CP"]

sns.countplot(absence, data=dataset,ax=axes[0]).set_title('Absence of Heart Disease')

sns.countplot(presence, data=dataset,ax=axes[1]).set_title('Presence of Heart Disease')

plt.show()

f.savefig('cp.png')
# Chi-square test of independence of variables

cont = pd.crosstab(dataset['CP'],dataset['Goal'])

chi_stat = stats.chi2_contingency(cont)

print(f'Chi statistics is {chi_stat[0]} and  p value is {chi_stat[1]}')

#Null hypothesis : Chest Pain is not associated with Goal

#Alternate hypothesis : Chest Pain is associated with Goal 
sizes = [len(dataset[dataset['FBS'] == 'No']), len(dataset[dataset['FBS'] == 'Yes'])]

labels = ['No', 'Yes']

plt.pie(x=sizes, labels=labels, explode=(0.1, 0), autopct="%1.2f%%", startangle=90,shadow=True)

plt.show()

# Fbs: fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
f, axes = plt.subplots(1,2,figsize=(15,5))

absence = dataset[dataset["Goal"]=='Absence']["FBS"]

presence = dataset[dataset["Goal"]=='Presence']["FBS"]

sns.countplot(absence, data=dataset,ax=axes[0]).set_title('Absence of Heart Disease')

sns.countplot(presence, data=dataset,ax=axes[1]).set_title('Presence of Heart Disease')

plt.show()

f.savefig('fbs.png')
# Chi-square test of independence of variables

cont = pd.crosstab(dataset['FBS'],dataset['Goal'])

chi_stat = stats.chi2_contingency(cont)

print(f'Chi statistics is {chi_stat[0]} and  p value is {chi_stat[1]}')

#Null hypothesis : FBS is not associated with Goal

#Alternate hypothesis : FBS is associated with Goal 
sizes = [len(dataset[dataset['RestECG'] =='Normal']), len(dataset[dataset['RestECG']=='Abnormality']), len(dataset[dataset['RestECG']=='Hypertrophy'])]

labels = ['Normal', 'ST-T wave abnormality', 'definite left ventricular hypertrophy by Estes criteria']

plt.pie(x=sizes, labels=labels, explode=(0, 0, 0), autopct="%1.2f%%", startangle=90,shadow=True)

plt.show()
f, axes = plt.subplots(1,2,figsize=(15,5))

absence = dataset[dataset["Goal"]=='Absence']["RestECG"]

presence = dataset[dataset["Goal"]=='Presence']["RestECG"]

sns.countplot(absence, data=dataset,ax=axes[0],order=['Normal', 'Abnormality', 'Hypertrophy']).set_title('Absence of Heart Disease')

sns.countplot(presence,ax=axes[1],order=['Normal', 'Abnormality', 'Hypertrophy']).set_title('Presence of Heart Disease')

plt.show()

f.savefig('restecg.png')
print(f'Probability of Hypertropy in disease cohorts {presence[presence=="Hypertrophy"].value_counts()/len(presence)}')

print(f'Probability of Hypertropy in non-disease cohorts {absence[absence=="Hypertrophy"].value_counts()/len(absence)}')

cont = pd.crosstab(dataset['RestECG'],dataset['Goal'])

chi_stat = stats.chi2_contingency(cont)

print(f'Chi statistics is {chi_stat[0]} and  p value is {chi_stat[1]}')

#Null hypothesis : Exang is not associated with Goal

#Alternate hypothesis : Exang is associated with Goal 
sns.countplot(data =dataset , x = 'Exang')

# exercise induced angina (1 = yes; 0 = no)
f, axes = plt.subplots(1,2,figsize=(15,5))

absence = dataset[dataset["Goal"]=='Absence']["Exang"]

presence = dataset[dataset["Goal"]=='Presence']["Exang"]

sns.countplot(absence, data=dataset,ax=axes[0]).set_title('Absence of Heart Disease')

sns.countplot(presence, data=dataset,ax=axes[1]).set_title('Presence of Heart Disease')

plt.show()

f.savefig('exang.png')
# Chi-square test of independence of variables

cont = pd.crosstab(dataset['Exang'],dataset['Goal'])

chi_stat = stats.chi2_contingency(cont)

print(f'Chi statistics is {chi_stat[0]} and  p value is {chi_stat[1]}')

#Null hypothesis : Exang is not associated with Goal

#Alternate hypothesis : Exang is associated with Goal 
sns.countplot(data =dataset , x = 'Slope')

# Slope: the slope of the peak exercise ST segment

    # Value 1: upsloping

    # Value 2: flat

    # Value 3: down-sloping
f, axes = plt.subplots(1,2,figsize=(15,5))

absence = dataset[dataset["Goal"]=='Absence']["Slope"]

presence = dataset[dataset["Goal"]=='Presence']["Slope"]

sns.countplot(absence, data=dataset,ax=axes[0]).set_title('Absence of Heart Disease')

sns.countplot(presence, data=dataset,ax=axes[1]).set_title('Presence of Heart Disease')

plt.show()

f.savefig('slope.png')
# Chi-square test of independence of variables

cont = pd.crosstab(dataset['Slope'],dataset['Goal'])

chi_stat = stats.chi2_contingency(cont)

print(f'Chi statistics is {chi_stat[0]} and  p value is {chi_stat[1]}')

#Null hypothesis : Slope is not associated with Goal

#Alternate hypothesis : Slope is associated with Goal 
sns.countplot(data =dataset , x = 'CA')
f, axes = plt.subplots(1,2,figsize=(15,5))

absence = dataset[dataset["Goal"]=='Absence']["CA"]

presence = dataset[dataset["Goal"]=='Presence']["CA"]

sns.countplot(absence, data=dataset,ax=axes[0]).set_title('Absence of Heart Disease')

sns.countplot(presence, data=dataset,ax=axes[1]).set_title('Presence of Heart Disease')

plt.show()

f.savefig('ca.png')
# Chi-square test of independence of variables

cont = pd.crosstab(dataset['CA'],dataset['Goal'])

chi_stat = stats.chi2_contingency(cont)

print(f'Chi statistics is {chi_stat[0]} and  p value is {chi_stat[1]}')

#Null hypothesis : CA is not associated with Goal

#Alternate hypothesis : CA is associated with Goal 
sizes = [len(dataset[dataset['Thal'] =='Normal']), len(dataset[dataset['Thal']=='Fixed Defect']), len(dataset[dataset['Thal']=='Reversible defect'])]

labels = ['Normal', 'Fixed Defect', 'Reversible defect']

plt.pie(x=sizes, labels=labels, explode=(0, 0, 0), autopct="%1.2f%%", startangle=90,shadow=True)

plt.show()
f, axes = plt.subplots(1,2,figsize=(15,5))

absence = dataset[dataset["Goal"]=='Absence']["Thal"]

presence = dataset[dataset["Goal"]=='Presence']["Thal"]

sns.countplot(absence, data=dataset,ax=axes[0],order=['Normal', 'Fixed Defect', 'Reversible defect']).set_title('Absence of Heart Disease')

sns.countplot(presence, data=dataset,ax=axes[1],order=['Normal', 'Fixed Defect', 'Reversible defect']).set_title('Presence of Heart Disease')

plt.show()

f.savefig('thal.png')
# Chi-square test of independence of variables

cont = pd.crosstab(dataset['Thal'],dataset['Goal'])

chi_stat = stats.chi2_contingency(cont)

print(f'Chi statistics is {chi_stat[0]} and  p value is {chi_stat[1]}')

#Null hypothesis : CA is not associated with Goal

#Alternate hypothesis : CA is associated with Goal 
sns.catplot(x="CP",hue="Goal", row="Gender",col="RestECG",data=dataset,kind="count",margin_titles=True)
sns.catplot(x="CP",hue="Goal", row="Gender",col="Exang",data=dataset, kind="count",margin_titles=True);

# Exang: exercise induced angina (1 = yes; 0 = no)
dataset['Goal']=dataset['Goal'].replace( ['Absence', 'Presence'],[0,1])

dataset['Goal']=dataset['Goal'].astype('int64')

dataset = pd.get_dummies(dataset,drop_first=False)
dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset)).values

dataset.head()
X_train, X_test, y_train, y_test = train_test_split(dataset.drop('Goal', 1), dataset['Goal'], test_size = .2, random_state=42,shuffle=True)
from sklearn.model_selection import GridSearchCV

lr = LogisticRegression(class_weight='balanced',random_state=42)

param_grid = { 

    'C': [0.1,0.2,0.3,0.4],

    'penalty': ['l1', 'l2'],

    'class_weight':[{0: 1, 1: 1},{ 0:0.67, 1:0.33 },{ 0:0.75, 1:0.25 },{ 0:0.8, 1:0.2 }]}

CV_rfc = GridSearchCV(estimator=lr, param_grid=param_grid, cv= 5)

CV_rfc.fit(X_train, y_train)

CV_rfc.best_params_
#fitting the model

lr1=LogisticRegression(C=0.2,random_state=42,penalty='l1',class_weight={0:1,1:1})

lr1.fit(X_train,y_train)
y_pred1=lr1.predict(X_test)

print("Logistic Train score with ",format(lr1.score(X_train, y_train)))

print("Logistic Test score with ",format(lr1.score(X_test, y_test)))
class_names=[0,1]

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

cm2 = confusion_matrix(y_test, y_pred1)

sns.heatmap(cm2, annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

sensitivity2 = cm2[1,1]/(cm2[1,1]+cm2[1,0])

print('Sensitivity/Recall : ', sensitivity2)



specificity2 = cm2[0,0]/(cm2[0,0]+cm2[0,1])

print('Specificity : ', specificity2)



precision2 = cm2[1,1]/(cm2[1,1]+cm2[0,1])

print('Precision   : ', precision2)



F1score2=(2*sensitivity2*precision2)/(sensitivity2+precision2)

print('F1 score    : ', F1score2)
test_attributes=X_test[['Oldpeak','Thalach','CA','Thal_Reversible defect','CP_Asymptomatic pain']]

train_attributes=X_train[['Oldpeak','Thalach','CA','Thal_Reversible defect','CP_Asymptomatic pain']]

lr2=LogisticRegression(C=0.3,penalty='l2',class_weight={0:1,1:1})

lr2.fit(train_attributes,y_train)
y_pred2=lr2.predict(test_attributes)

print("Logistic Train score with ",format(lr2.score(train_attributes, y_train)))

print("Logistic Test score with ",format(accuracy_score(y_pred1, y_test)))
class_names=[0,1]

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

cm2 = confusion_matrix(y_test, y_pred2)

sns.heatmap(cm2, annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

sensitivity2 = cm2[1,1]/(cm2[1,1]+cm2[1,0])

print('Sensitivity/Recall : ', sensitivity2)



specificity2 = cm2[0,0]/(cm2[0,0]+cm2[0,1])

print('Specificity : ', specificity2)



precision2 = cm2[1,1]/(cm2[1,1]+cm2[0,1])

print('Precision   : ', precision2)



F1score2=(2*sensitivity2*precision2)/(sensitivity2+precision2)

print('F1 score    : ', F1score2)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

skf.get_n_splits(X_train, y_train)

results = cross_val_score(lr1, X_train, y_train, cv=skf, n_jobs=1, scoring='accuracy')

results.mean()
perm_imp1 = PermutationImportance(lr1, random_state=42,scoring='accuracy').fit(X_test, y_test)

eli5.show_weights(perm_imp1, feature_names = X_test.columns.tolist(),top=50)
X_train2=X_train.drop(['Exang_No'],axis=1)

X_test2=X_test.drop(['Exang_No'],axis=1)
lr2=LogisticRegression(C=0.2,penalty='l1',class_weight={0:1,1:1})

lr2.fit(X_train2,y_train)

y_pred2=lr2.predict(X_test2)
print("Logistic TRAIN score with ",format(lr2.score(X_train2, y_train)))

print("Logistic TEST score with ",format(lr2.score(X_test2, y_test)))
perm_imp1 = PermutationImportance(lr2, random_state=42,scoring='accuracy').fit(X_test2, y_test)

eli5.show_weights(perm_imp1, feature_names = X_test2.columns.tolist(),top=50)
X_train3=X_train2.drop(['Age','Thal_Fixed Defect','Slope_Flat','Slope_Down-sloping','RestECG_Hypertrophy','RestECG_Abnormality','FBS_Yes','Oldpeak'],axis=1)

X_test3=X_test2.drop(['Age','Thal_Fixed Defect','Slope_Flat','Slope_Down-sloping','RestECG_Hypertrophy','RestECG_Abnormality','FBS_Yes','Oldpeak'],axis=1)
lr3=LogisticRegression(C=0.3,penalty='l2',class_weight={0:1,1:1})

lr3.fit(X_train3,y_train)

y_pred3=lr3.predict(X_test3)

y_probab3=lr3.predict_proba(X_test3)
print("Logistic TRAIN score with ",format(lr2.score(X_train2, y_train)))

print("Logistic TEST score with ",format(lr2.score(X_test2, y_test)))
class_names=[0,1]

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

cm2 = confusion_matrix(y_test, y_pred3)

sns.heatmap(cm2, annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

ax.yaxis.set_label_position("right")

plt.tight_layout()

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.savefig("cmlr.png")
sensitivity2 = cm2[1,1]/(cm2[1,1]+cm2[1,0])

print('Sensitivity/Recall : ', sensitivity2)



specificity2 = cm2[0,0]/(cm2[0,0]+cm2[0,1])

print('Specificity : ', specificity2)



precision2 = cm2[1,1]/(cm2[1,1]+cm2[0,1])

print('Precision   : ', precision2)



F1score2=(2*sensitivity2*precision2)/(sensitivity2+precision2)

print('F1 score    : ', F1score2)
fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred2)

fig, ax = plt.subplots()

ax.plot(fpr1, tpr1)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve for diabetes classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)

plt.savefig("roclr.png")
#Higher the AUC, better the model is at distinguishing between patients with disease and no disease.

roc_auc2 = auc(fpr1, tpr1)

roc_auc2
y_pred04=[]

for i in range(len(y_probab3)):

    y_pred04.append(1 if y_probab3[i,1]>0.3 else 0)

class_names=[0,1]

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

cm04 = confusion_matrix(y_test, y_pred04)

sns.heatmap(cm04, annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
sensitivity04 = cm04[1,1]/(cm04[1,1]+cm04[1,0])

print('Sensitivity/Recall : ', sensitivity04)



specificity04 = cm04[0,0]/(cm04[0,0]+cm04[0,1])

print('Specificity : ', specificity04)



precision04 = cm04[1,1]/(cm04[1,1]+cm04[0,1])

print('Precision   : ', precision04)



F1score04=(2*sensitivity04*precision04)/(sensitivity04+precision04)

print('F1 score    : ', F1score04)
from sklearn.decomposition import PCA

pca = PCA(n_components=10)

pca.fit(X_train)

trainComponents = pca.fit_transform(X_train)

testComponents = pca.fit_transform(X_test)

lr=LogisticRegression(C=0.3,penalty='l2',class_weight={0:1,1:1})

lr.fit(trainComponents,y_train)

y_pred5=lr.predict(testComponents)

print("Logistic TRAIN score with ",format(lr.score(trainComponents, y_train)))

print("Logistic TEST score with ",format(lr.score(testComponents, y_test)))
pca.explained_variance_ratio_.cumsum()
class_names=[0,1]

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

cm04 = confusion_matrix(y_test, y_pred5)

sns.heatmap(cm04, annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
sensitivity2 = cm2[1,1]/(cm2[1,1]+cm2[1,0])

print('Sensitivity/Recall : ', sensitivity2)



specificity2 = cm2[0,0]/(cm2[0,0]+cm2[0,1])

print('Specificity : ', specificity2)



precision2 = cm2[1,1]/(cm2[1,1]+cm2[0,1])

print('Precision   : ', precision2)



F1score2=(2*sensitivity2*precision2)/(sensitivity2+precision2)

print('F1 score    : ', F1score2)
from sklearn.model_selection import GridSearchCV

rfc = RandomForestClassifier(oob_score=True,random_state=42)

param_grid = { 

    'n_estimators': [200,300,500],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6],

    'criterion' :['gini', 'entropy']

}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 3)

CV_rfc.fit(X_train, y_train)

CV_rfc.best_params_
model = RandomForestClassifier(max_depth=6,oob_score=True,random_state=42,criterion='entropy',max_features='auto',n_estimators=300)

model.fit(X_train, y_train)
estimator = model.estimators_[3]

feature_names = [i for i in X_train.columns]



y_train_str = y_train.astype('str')

y_train_str[y_train_str == '0'] = 'Absence'

y_train_str[y_train_str == '1'] = 'Presence'

y_train_str = y_train_str.values
export_graphviz(estimator, out_file='tree.dot', 

                feature_names = feature_names,

                class_names = y_train_str,

                rounded = True, proportion = True, 

                label='root',

                precision = 2, filled = True)



import os

os.environ["PATH"] += os.pathsep + 'C:\\Users\\u22v03\\Documents\\Python Scripts\\heart\\release\\bin'

!dot -Tpng tree.dot -o tree.png -Gdpi=600



from IPython.display import Image

Image(filename = 'tree.png')
y_pred_quant = model.predict_proba(X_test)[:, 1]

y_pred_bin = model.predict(X_test)
print("Random forest TRAIN score with ",format(model.score(X_train, y_train)))

print("Random forest TEST score with ",format(model.score(X_test, y_test)))
feature_importances = pd.DataFrame(model.feature_importances_,

                                   index = X_train.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)

feature_importances
cm4 = confusion_matrix(y_test, y_pred_bin)

class_names=[0,1]

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

sns.heatmap(cm4,annot=True)

ax.xaxis.set_label_position("top")

ax.yaxis.set_label_position("right")

plt.tight_layout()

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.savefig("cmrf.png")
sensitivity4 = cm4[1,1]/(cm4[1,1]+cm4[1,0])#how good a test is at detecting the positives

print('Sensitivity/Recall : ', sensitivity4)



specificity4 = cm4[0,0]/(cm4[0,0]+cm4[0,1])#how good a test is at avoiding false alarms

print('Specificity : ', specificity4)



precision4 = cm4[1,1]/(cm4[1,1]+cm4[0,1])#how many of the positively classified were relevant

print('Precision   : ', precision4)



F1score4=(2*sensitivity4*precision4)/(sensitivity4+precision4)# low false positives and low false negatives

print('F1 score    : ', F1score4)
fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred_bin)

fig, ax = plt.subplots()

ax.plot(fpr1, tpr1)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve for diabetes classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)

plt.savefig("rocrf.png")
X_train5=X_train.drop(['RestECG_Abnormality'],axis=1)

X_test5=X_test.drop(['RestECG_Abnormality'],axis=1)
model = RandomForestClassifier(max_depth=5,oob_score=True,random_state=42,criterion='gini',max_features='auto',n_estimators=300)

model.fit(X_train5, y_train)
y_pred_quant = model.predict_proba(X_test5)[:, 1]

y_pred_bin = model.predict(X_test5)
print("Random forest TRAIN score with ",format(model.score(X_train5, y_train)))

print("Random forest TEST score with ",format(model.score(X_test5, y_test)))
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred_bin=model.predict(X_test)
param_grid = {"criterion": ['entropy', 'gini'],

              "min_samples_split": [5,10,15],

              "max_depth": [2,3,5],

              "min_samples_leaf": [5,10,15],

              "max_leaf_nodes": [5,10,15],

              }



CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv= 3)

CV_rfc.fit(X_train, y_train)

CV_rfc.best_params_
print("Random forest TRAIN score with ",format(model.score(X_train, y_train)))

print("Random forest TEST score with ",format(model.score(X_test, y_test)))