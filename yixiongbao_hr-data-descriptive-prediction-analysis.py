import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
HR = pd.read_csv("../input/HR_comma_sep.csv")
HR.shape
HR.dtypes
HR.corr()
corr = HR.corr()

plt.figure(figsize=(8,8))

sns.heatmap(corr,vmax=1,vmin=-1,square=True,annot=True)
HR['Current_Status'] = HR['left'].apply(lambda x: 'Stay' if x == 0 else 'Leave')

HR.groupby(['sales','salary','Current_Status']).size()
leave = HR[(HR['left'] == 1)]

stay = HR[(HR['left']==0)]

dept_name = HR['sales'].unique()

name=['Sales','Accounting','HR','Technical','Support','Management','IT','Product Management','Marketing','RandD']

index = range(10)



plt.figure(1,figsize=(12,8))



plt.subplot(1,2,1)

leave['sales'].value_counts().plot(kind='bar')

plt.title('Employess who LEAVE the company by department')

plt.xticks(index,name)





plt.subplot(1,2,2)

stay['sales'].value_counts().plot(kind='bar',color='green')

plt.title('Employess who STAY in the company by department')

plt.xticks(index,name)
L_salary_level_count = leave['salary'].value_counts()

S_salary_level_count = stay['salary'].value_counts()



plt.figure(1,figsize=(12,8))

plt.subplot(1,2,1)

L_salary_level_count.plot(kind='bar',rot=0)

plt.title('The number of employees who \n LEAVE the company by salary level ')



plt.subplot(1,2,2)

S_salary_level_count.plot(kind='bar',rot=0,color='green')

plt.title('The number of employees who \n STAY the company by salary level ')
def plot_dept_leave_salary(department):

    dept_leave = leave[leave['sales'] ==department]

    count = dept_leave['salary'].value_counts()

    index = [1,2,3]

#     color = ['red','blue','green']

    plt.bar(index,count,width=0.5)

    plt.xticks(index,['Low','Medium','High'])

    

def plot_dept_stay_salary(department):

    dept_stay = stay[stay['sales'] ==department]

    count = dept_stay['salary'].value_counts()

    index = [1,2,3]

    color = ['red','blue','green']

    plt.bar(index,count,width=0.5,color='green')

    plt.xticks(index,['Low','Medium','High'])
plt.figure(1,figsize=(12,8))

for i in range(10):

    plt.subplot(2,5,i+1)

    plot_dept_leave_salary(dept_name[i])

    plt.title(name[i])

plt.suptitle('LEAVE')
plt.figure(1,figsize=(12,8))

for i in range(10):

    plt.subplot(2,5,i+1)

    plot_dept_stay_salary(dept_name[i])

    plt.title(name[i])

plt.suptitle('STAY')
leave_time_spend_count = leave['time_spend_company'].value_counts().sort_index()

stay_time_spend_count = stay['time_spend_company'].value_counts().sort_index()



plt.figure(1,figsize=(12,8))

plt.subplot(1,2,1)

leave_time_spend_count.plot(kind='bar',rot=0)

plt.title('How many years did they spend in the \n company before LEAVING')

plt.xlabel('Years')

plt.ylabel('Number of employees')



plt.subplot(1,2,2)

stay_time_spend_count.plot(kind='bar',rot=0,color='green')

plt.title('How many years did they spend in the \n company continue STAYING')

plt.xlabel('Years')

plt.ylabel('Number of employees')
def plot_dept_leave_years(department):

    year_leave = leave[leave['sales'] ==department]

    count = year_leave['time_spend_company'].value_counts().sort_index()

    index = range(0,len(count))

    check = year_leave['time_spend_company'].unique()

    plt.bar(index,count,width=0.5)

    plt.xticks(index,sorted(check))



def plot_dept_stay_years(department):

    year_stay = stay[stay['sales'] ==department]

    count = year_stay['time_spend_company'].value_counts().sort_index()

    index = range(0,len(count))

    check = year_stay['time_spend_company'].unique()

    plt.bar(index,count,width=0.5,color='green')

    plt.xticks(index,sorted(check))
plt.figure(1,figsize=(12,8))

for i in range(10):

    plt.subplot(2,5,i+1)

    plot_dept_leave_years(dept_name[i])

    plt.title(name[i])

plt.suptitle('LEAVE')
plt.figure(1,figsize=(12,8))

for i in range(10):

    plt.subplot(2,5,i+1)

    plot_dept_stay_years(dept_name[i])

    plt.title(name[i])

plt.suptitle('STAY')
del HR['sales']

del HR['salary']

del HR['Current_Status']
from sklearn.model_selection import train_test_split

label = HR.pop('left')

HR_train,HR_test,label_train,label_test = train_test_split(HR, label, test_size = 0.2, random_state = 42)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB
Classifiers = [

    LogisticRegression(),

    KNeighborsClassifier(),

    SVC(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GaussianNB()]



def run_classifiers(HR_train,label_train,HR_test,label_test):

    Training_score = []

    Testing_score = []

    Model = []

    Accuracy = []

    for classifier in Classifiers:

        Model.append(classifier.__class__.__name__)

        classifier.fit(HR_train,label_train)

        Trs = classifier.score(HR_train,label_train)

        Training_score.append(Trs)

        Tes = classifier.score(HR_test,label_test)

        Testing_score.append(Tes)

    

        print ("The "+Model[i]+" has a training score of "+ \

        Training_score[i]+" and testing score of "+Testing_score[i])
Training_score = []

Testing_score = []

Model = []

def run_classifiers(HR_train,label_train,HR_test,label_test):

    Training_score = []

    Testing_score = []

    Model = []

    Accuracy = []

    for classifier in Classifiers:

        Model.append(classifier.__class__.__name__)

        classifier.fit(HR_train,label_train)

        Training_score.append(classifier.score(HR_train,label_train))

        Testing_score.append(classifier.score(HR_test,label_test))

    

#    for i in range(7):

#        print "The "+str(Model[i])+" has a training score of "+str(Training_score[i])+" and testing score of "+str(Testing_score[i])

### not familar the print method in Python3.6, but it works well on my laptop with Python2.7;         
