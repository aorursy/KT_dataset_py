# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
heart_study = pd.read_csv('/kaggle/input/framingham-heart-study-dataset/framingham.csv')

heart_study.head()
heart_study.dtypes

heart_study.isnull().sum()

heart_study.describe()
#cigsperday all currentSmoker
heart_study[heart_study.cigsPerDay.isna()].currentSmoker.unique()
heart_study.corr()
# cols that contains null values: 
#education,cigsPerDay(FILL MEAN BY male or female),BPMeds(fill with most frequent for each age range),totChol(fill with most frequent for each age range),BMI(fill with most frequent for each age range),heartRate(by gender most frequent),glucose
heart_study.corr()[['education','cigsPerDay','BPMeds','totChol','BMI','heartRate','glucose','TenYearCHD']]
heart_study[heart_study.prevalentHyp==1].describe()
%matplotlib inline
heart_study.hist(bins=20)
sns.pairplot(data = heart_study[['education','age','totChol','sysBP','BMI','TenYearCHD']] )
heart_study.age.hist()
def change_age_to_range(x):
    if x<=34:
        return 0
    elif x<=55:
        return 1
    else:
        return 2
heart_study['age_range'] = heart_study.age.map(change_age_to_range)

##education,cigsPerDay(FILL MEAN BY male or female),BPMeds(fill with most frequent for each age range),totChol(fill with most frequent for each age range),BMI(fill with most frequent for each age range),heartRate(by gender most frequent),glucose


heart_study.groupby(['male','age_range']).agg({'cigsPerDay':'mean','heartRate':'mean','BMI':'mean','totChol':'mean','glucose':'mean'})
mean_dict = heart_study.groupby(['male','age_range']).agg({'cigsPerDay':'mean','heartRate':'mean','BMI':'mean','totChol':'mean','glucose':'mean'}).T.to_dict()
def map_mean_dict(x,col):
    male = x['male']
    age_range=x['age_range']
    if str(x[col])=='nan':
        return mean_dict[(male,age_range)][col]
    else:
        return x[col]
na_cols = ['cigsPerDay','heartRate', 'BMI', 'totChol', 'glucose']
for col in na_cols:
    heart_study[col] = heart_study.apply(lambda x:map_mean_dict(x,col),axis=1)
heart_study['BPMeds'].fillna(0,inplace=True)
heart_study['current_glucose_high']=heart_study['glucose'].apply(lambda x:1 if x>=125 else 0)

heart_study['current_BMI_low']=heart_study['BMI'].apply(lambda x:1 if x<18.5 else 0)
heart_study['current_BMI_normal']=heart_study['BMI'].apply(lambda x:1 if ((x>=18.5) and (x<=24.9)) else 0)
heart_study['current_BMI_high']=heart_study['BMI'].apply(lambda x:1 if  x>24.9 else 0)
def cholesterol_level(totChol):
    if totChol<200:
        return 0
    elif totChol<=239:
        return 1
    else:
        return 2
heart_study['chol_level'] = heart_study['totChol'].map(cholesterol_level)
def get_current_bp_level(x):
    sysBP = x['sysBP']
    diaBP = x['diaBP']
    if (sysBP<120) and (diaBP<80):
        return 0
    elif (sysBP<=129) and (diaBP<80):
        return 1
    elif (sysBP<=139) or ((diaBP<=89) and (diaBP>=80) ):
        return 2
    elif (sysBP>139) or (diaBP>=90):
        return 3
    elif (sysBP>=180) or (diaBP>=120):
        return 4
    else:
        print(sysBP,diaBP)
        return None
heart_study['current_bp_level'] = heart_study.apply(lambda x: get_current_bp_level(x),axis=1)
heart_study['heartRate_low'] = heart_study.heartRate.apply(lambda x:1 if x<60 else 0)
heart_study['heartRate_normal'] = heart_study.heartRate.apply(lambda x:1 if ((x>=60) and (x<=100)) else 0)
heart_study['heartRate_high'] = heart_study.heartRate.apply(lambda x:1 if x>100 else 0)




heart_study.head()

# #covert
convert_cols=[ 'age', 'cigsPerDay', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose']
need_convert = heart_study[convert_cols]
need_convert = need_convert.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
heart_study[convert_cols]=need_convert
selected_df = pd.concat([heart_study[heart_study.TenYearCHD==0].sample(644),heart_study[heart_study.TenYearCHD==1]])
selected_df['had_disease'] = selected_df['prevalentStroke']|selected_df['prevalentHyp']|selected_df['diabetes']
selected_df.corr()[['TenYearCHD']]
corr = selected_df.corr()[['TenYearCHD']]
corr
needed_cols=['male', 'age',  'prevalentHyp',  'prevalentStroke', 'sysBP',
       'current_glucose_high','diabetes',
           'had_disease']

for_model = selected_df[needed_cols]


y = selected_df['TenYearCHD'].values
x_train,x_test,y_train,y_test=train_test_split(for_model.values,y,test_size=.20,random_state=32)
normal = LogisticRegression()
normal.fit(x_train, y_train)
normal_ypred_prob = normal.predict_proba(x_test)
normal_y_pred =  normal.predict(x_test)
print("{:<40} {:.2f}%".format("AUC normal:", 100*roc_auc_score(y_test,normal_ypred_prob[:,1])))

cm=confusion_matrix(y_test,normal_y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d')
fpr, tpr, thresholds = roc_curve(y_test, normal_ypred_prob[:,1])
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Heart disease classifier')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)


# Set regularization parameter
for i, t in enumerate(range(0,10)):

    # turn down tolerance for short training time
    clf_l1_LR = LogisticRegression(penalty='l1', tol=t*0.01, solver='saga',max_iter=2000)
    clf_l2_LR = LogisticRegression(penalty='l2', tol=t*0.01, solver='saga',max_iter=2000)

    clf_l1_LR.fit(x_train, y_train)
    clf_l2_LR.fit(x_train, y_train)

   
    l1_y_pred_prob =  clf_l1_LR.predict_proba(x_test)
    l2_y_pred_prob =  clf_l2_LR.predict_proba(x_test)

    # coef_l1_LR contains zeros due to the
    # L1 sparsity inducing norm

    t=t*0.01

    print("t=%.2f" % t)
    
    print("{:<40} {:.2f}".format("Score with L1 penalty:",
                                 clf_l1_LR.score(x_test, y_test)))

    print("{:<40} {:.2f}".format("Score with L2 penalty:",
                                 clf_l2_LR.score(x_test, y_test)))
    print("{:<40} {:.2f}%".format("AUC with L1 penalty:", 100*roc_auc_score(y_test,l1_y_pred_prob[:,1])))
    print("{:<40} {:.2f}%".format("AUC with L2 penalty:", 100*roc_auc_score(y_test,l2_y_pred_prob[:,1])))
     


