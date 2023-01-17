import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.svm import LinearSVC
df=pd.read_csv('../input/bank-marketing/bank-additional-full.csv',delimiter=';')
df1=df

df1.columns
df1.info()
df1.describe()
df1['target']=df1.y.apply(lambda x:1 if x=='yes' else 0)
df1.drop('y',axis=1,inplace=True)
corr=df1.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr,vmax=.3, square=True,annot=True)

df1.isna().sum()
#bins = np.linspace(min(df1["age"]), max(df1["age"]), 4)
#group_names = ['Young', 'Middle_Aged', 'Old']
#df1['Age_Cat'] = pd.cut(df1['age'], bins, labels=group_names, include_lowest=True )
#df1[['age','Age_Cat']].head(20)
df1['age_cat']=df1.age.apply(lambda x: 'Young' if 18 <= x <30 else('Middle_Aged' if 30 <= x < 50 else 'Old'))
dummy_variable_1 = pd.get_dummies(df1["age_cat"])
df1 = pd.concat([df1, dummy_variable_1], axis=1)
#df1['job'].value_counts()
dummy_variable_2 = pd.get_dummies(df1["job"])
df1 = pd.concat([df1, dummy_variable_2], axis=1)
df1.rename(columns={'unknown':'unknown_job'},inplace=True)
#df1.drop("unknown", axis = 1, inplace=True)
#df1['marital'].value_counts()
dummy_variable_3 = pd.get_dummies(df1["marital"])
df1 = pd.concat([df1, dummy_variable_3], axis=1)
df1.rename(columns={'unknown':'unknown_marital_status'},inplace=True)
#df1.drop("unknown", axis = 1, inplace=True)
#df1['education'].value_counts()
dummy_variable_4 = pd.get_dummies(df1["education"])
df1 = pd.concat([df1, dummy_variable_4], axis=1)
df1.rename(columns={'unknown':'unknown_education'},inplace=True)
#df1.drop("unknown", axis = 1, inplace=True)
#df1['housing'].value_counts()
dummy_variable_12 = pd.get_dummies(df1["housing"])
dummy_variable_12.rename(columns={'no':'No_House', 'yes':'Has_House','unknown':'unknown_housing_status'}, inplace=True)
df1 = pd.concat([df1, dummy_variable_12], axis=1)
#df1.drop("Unknown", axis = 1, inplace=True)

#df1['default'].value_counts()
dummy_variable_5 = pd.get_dummies(df1["default"])
dummy_variable_5.rename(columns={'no':'no_default_credit', 'yes':'has_default_credit','unknown':'unknown_default_credit'}, inplace=True)
df1 = pd.concat([df1, dummy_variable_5], axis=1)
#df1.drop("Unknown", axis = 1, inplace=True)

#df1['loan'].value_counts()
dummy_variable_6 = pd.get_dummies(df1["loan"])
dummy_variable_6.rename(columns={'no':'Loan_Not_Taken', 'yes':'Loan_Taken','unknown':'unknown_loan_status'}, inplace=True)
df1 = pd.concat([df1, dummy_variable_6], axis=1)
#df1.drop("Unknown", axis = 1, inplace=True)

#df1['contact'].value_counts()
dummy_variable_7 = pd.get_dummies(df1["contact"])
df1 = pd.concat([df1, dummy_variable_7], axis=1)
#df1['month'].value_counts()
dummy_variable_8 = pd.get_dummies(df1["month"])
df1 = pd.concat([df1, dummy_variable_8], axis=1)
#df1['day_of_week'].value_counts()
dummy_variable_9 = pd.get_dummies(df1["day_of_week"])
df1 = pd.concat([df1, dummy_variable_9], axis=1)
#df1['campaign'].value_counts()
df1['campaign_cat']=df1.campaign.apply(lambda x: 'Few_times' if 0 <= x <15 else('Many_times' if 15 <= x < 30 else 'Alot_of_Times'))
dummy_variable_10 = pd.get_dummies(df1["campaign_cat"])
df1 = pd.concat([df1, dummy_variable_10], axis=1)
#df1['pdays'].value_counts()
df1['not_contacted_previously']=df1.pdays.apply(lambda x: 1 if x == 999 else 0)
df1['one_week_till_last_contact']=df1.pdays.apply(lambda x: 1 if 0<=x<=7 else 0)
df1['two_weeks_till_last_contact']=df1.pdays.apply(lambda x: 1 if 7<x<=14 else 0)
#df1['previous'].value_counts()
df1['zero_times_contacted']=df1.previous.apply(lambda x: 1 if x == 0 else 0)
df1['one_time_contacted']=df1.previous.apply(lambda x: 1 if x == 1 else 0)
df1['two_times_contacted']=df1.previous.apply(lambda x: 1 if x == 2 else 0)
df1['three_times_contacted']=df1.previous.apply(lambda x: 1 if x == 3 else 0)
df1['four_times_contacted']=df1.previous.apply(lambda x: 1 if x == 4 else 0)
df1['poutcome'].value_counts()
dummy_variable_11 = pd.get_dummies(df1["poutcome"])
dummy_variable_11.rename(columns={'nonexistent':'Not_campaigned', 'success':'previous_campaign_successful','failure':'previous_campaign_failure'}, inplace=True)
df1 = pd.concat([df1, dummy_variable_11], axis=1)


dff=df1[['emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']]
cor=dff.corr()
plt.figure(figsize=(8,6))
sns.heatmap(cor,vmax=.3, square=True,annot=True)

#!pip install factor_analyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(dff)
p_value
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(dff)
kmo_model
Factor=FactorAnalysis(n_components=1)
df1['Fact']=Factor.fit_transform(dff)
df1.drop(['emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed'],axis=1,inplace=True)
#df1.drop(['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','age_cat','campaign_cat'],axis=1,inplace=True)
df1.columns 
#scaler=preprocessing.MinMaxScaler()
#scaled_df=pd.DataFrame(scaler.fit_transform(df1),columns=df1.columns)
ax=sns.violinplot()
sns.violinplot(data=df1,x='age_cat',y='age')
ax.set_title("Ages Distribution")
ax.set_xlabel("Group of Ages")
ax.set_ylabel( "Distribution")
plt.show()
ax  = plt.subplot()
sns.kdeplot(df1['age'], shade=True)
ax.set_title("Age Distribution")
ax.set_xlabel("Ages")
ax.set_ylabel( "Plot")
plt.legend()

plt.show()

Gr=df1.groupby(['age_cat'],as_index=False)['age'].count()
Gr.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=Gr, x='age_cat', y = "Count")
    
    
dfsvt=df1[['age_cat','target','age']]
SVT=dfsvt.groupby(['age_cat','target'],as_index=False).count()
SVT.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=SVT, x="age_cat", y = "Count", hue="target")
plt.show()

plt.figure(figsize=(16,8))
Grj=df1.groupby(['job'],as_index=False)['age'].count()
Grj.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=Grj, x='job', y = "Count")
plt.show()   
plt.figure(figsize=(16,8))
dfsvt=df1[['job','target','age']]
SVT=dfsvt.groupby(['job','target'],as_index=False).count()
SVT.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=SVT, x="job", y = "Count", hue="target")
plt.show()
Grm=df1.groupby(['marital'],as_index=False)['age'].count()
Grm.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=Grm, x='marital', y = "Count")
plt.show()
dfsvt=df1[['marital','target','age']]
SVT=dfsvt.groupby(['marital','target'],as_index=False).count()
SVT.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=SVT, x="marital", y = "Count", hue="target")
plt.show()
plt.figure(figsize=(14,8))
Gre=df1.groupby(['education'],as_index=False)['age'].count()
Gre.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=Gre, x='education', y = "Count")
plt.show()
plt.figure(figsize=(14,8))
dfsvt=df1[['education','target','age']]
SVT=dfsvt.groupby(['education','target'],as_index=False).count()
SVT.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=SVT, x="education", y = "Count", hue="target")
plt.show()
Grd=df1.groupby(['default'],as_index=False)['age'].count()
Grd.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=Grd, x='default', y = "Count")
plt.show()
dfsvt=df1[['default','target','age']]
SVT=dfsvt.groupby(['default','target'],as_index=False).count()
SVT.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=SVT, x="default", y = "Count", hue="target")
plt.show()
Grh=df1.groupby(['housing'],as_index=False)['age'].count()
Grh.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=Grh, x='housing', y = "Count")
plt.show()
dfsvt=df1[['housing','target','age']]
SVT=dfsvt.groupby(['housing','target'],as_index=False).count()
SVT.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=SVT, x="housing", y = "Count", hue="target")
plt.show()
Grl=df1.groupby(['loan'],as_index=False)['age'].count()
Grl.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=Grl, x='loan', y = "Count")
plt.show()
dfsvt=df1[['loan','target','age']]
SVT=dfsvt.groupby(['loan','target'],as_index=False).count()
SVT.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=SVT, x="loan", y = "Count", hue="target")
plt.show()
Grc=df1.groupby(['contact'],as_index=False)['age'].count()
Grc.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=Grc, x='contact', y = "Count")
plt.show()
dfsvt=df1[['contact','target','age']]
SVT=dfsvt.groupby(['contact','target'],as_index=False).count()
SVT.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=SVT, x="contact", y = "Count", hue="target")
plt.show()
f, axes = plt.subplots(3, 3, figsize=(28, 14))
plt.suptitle('Barplot of months vs. their count')
Mon=['apr', 'aug', 'dec', 'jul','jun', 'mar', 'may', 'nov', 'oct']
for i, e in enumerate(Mon):
    Gr=df1.groupby([e],as_index=False)['age'].count()
    Gr.rename(columns={'age':'Count'},inplace=True)
    sns.barplot(data=Gr, x=e, y = "Count",ax=axes[i //3 ][i % 3 ])
f, axes = plt.subplots(3, 3, figsize=(28, 14))
plt.suptitle('Barplot of months vs. their target_count')
Mon=['apr', 'aug', 'dec', 'jul','jun', 'mar', 'may', 'nov', 'oct']
for i, e in enumerate(Mon):
    SVT=df1.groupby([e,'target'],as_index=False).count()
    SVT.rename(columns={'age':'Count'},inplace=True)
    sns.barplot(data=SVT, x=e, y = "Count", hue="target",ax=axes[i //3 ][i % 3 ])

plt.figure(figsize=(12,8))
Grd=df1.groupby(['day_of_week'],as_index=False)['age'].count()
Grd.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=Grd, x='day_of_week', y = "Count")
plt.show()
plt.figure(figsize=(12,8))
dfsvt=df1[['day_of_week','target','age']]
SVT=dfsvt.groupby(['day_of_week','target'],as_index=False).count()
SVT.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=SVT, x="day_of_week", y = "Count", hue="target")
plt.show()
camp=df1['campaign'].value_counts()
plt.plot(camp)

plt.title('Different number of times a person was Contacted')
plt.ylabel('Count')
plt.xlabel('Number of times contact was made')

# annotate the 2010 Earthquake. 
# syntax: plt.text(x, y, label)
#plt.text('Count of Contacts') # see note below

plt.show() 
Grc=df1.groupby(['campaign_cat'],as_index=False)['age'].count()
Grc.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=Grc, x='campaign_cat', y = "Count")
plt.show()
dfsvt=df1[['campaign_cat','target','age']]
SVT=dfsvt.groupby(['campaign_cat','target'],as_index=False).count()
SVT.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=SVT, x="campaign_cat", y = "Count", hue="target")
plt.show()
plt.figure(figsize=(16,6))
plt.suptitle('Days Passed vs. their count')


ax1  = plt.subplot(1,3,1)
Gr1=df1.groupby(['not_contacted_previously'],as_index=False)['age'].count()
Gr1.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=Gr1, x='not_contacted_previously', y = "Count")
ax2  = plt.subplot(1,3,2)
Gr2=df1.groupby(['one_week_till_last_contact'],as_index=False)['age'].count()
Gr2.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=Gr2, x='one_week_till_last_contact', y = "Count")
ax3  = plt.subplot(1,3,3)
Gr3=df1.groupby(['two_weeks_till_last_contact'],as_index=False)['age'].count()
Gr3.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=Gr3, x='two_weeks_till_last_contact', y = "Count")
plt.show()
   
   
plt.suptitle('Days Passed vs. their target count')
plt.figure(figsize=(16,6))

ax1  = plt.subplot(1,3,1)
Gr4=df1.groupby(['not_contacted_previously','target'],as_index=False)['age'].count()
Gr4.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=Gr4, x='not_contacted_previously', y = "Count",hue='target')
ax2  = plt.subplot(1,3,2)
Gr5=df1.groupby(['one_week_till_last_contact','target'],as_index=False)['age'].count()
Gr5.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=Gr5, x='one_week_till_last_contact', y = "Count",hue='target')
ax3  = plt.subplot(1,3,3)
Gr6=df1.groupby(['two_weeks_till_last_contact','target'],as_index=False)['age'].count()
Gr6.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=Gr6, x='two_weeks_till_last_contact', y = "Count",hue='target')
plt.show()
   
   

previous_contacts=['zero_times_contacted','one_time_contacted', 'two_times_contacted', 'three_times_contacted','four_times_contacted']
plt.suptitle('Previously Contacted vs. their count')
for i, e in enumerate(previous_contacts):
    Gr=df1.groupby([e],as_index=False)['age'].count()
    Gr.rename(columns={'age':'Count'},inplace=True)
    sns.barplot(data=Gr, x=e, y = "Count")
    plt.show()
previous_contacts=['zero_times_contacted','one_time_contacted', 'two_times_contacted', 'three_times_contacted','four_times_contacted']
plt.suptitle('Previously Contacted vs. their target count')
for i, e in enumerate(previous_contacts):
    Gr=df1.groupby([e,'target'],as_index=False)['age'].count()
    Gr.rename(columns={'age':'Count'},inplace=True)
    sns.barplot(data=Gr, x=e, y = "Count",hue='target')
    plt.show()
Grp=df1.groupby(['poutcome'],as_index=False)['age'].count()
Grp.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=Grp, x='poutcome', y = "Count")
plt.show()
dfsvt=df1[['poutcome','target','age']]
SVT=dfsvt.groupby(['poutcome','target'],as_index=False).count()
SVT.rename(columns={'age':'Count'},inplace=True)
sns.barplot(data=SVT, x="poutcome", y = "Count", hue="target")
plt.show()
sns.kdeplot(df1['Fact'], shade=True)
plt.legend()

plt.show()
df1.drop(['age_cat','job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','campaign_cat','duration','not_contacted_previously','pdays'],axis=1,inplace=True)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
#from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import jaccard_score
scaler=preprocessing.MinMaxScaler()
scaled=pd.DataFrame(scaler.fit_transform(df1),columns=df1.columns)
X=scaled[['age', 'campaign', 'previous', 'Middle_Aged', 'Old','Young', 'admin.', 'blue-collar', 'entrepreneur', 'housemaid','management', 'retired', 'self-employed', 'services', 'student','technician', 'unemployed', 'unknown_job', 'divorced', 'married','single', 'unknown_marital_status', 'basic.4y', 'basic.6y', 'basic.9y','high.school', 'illiterate', 'professional.course', 'university.degree','unknown_education', 'No_House', 'unknown_housing_status', 'Has_House','no_default_credit', 'unknown_default_credit', 'has_default_credit','Loan_Not_Taken', 'unknown_loan_status', 'Loan_Taken', 'cellular','telephone', 'apr', 'aug', 'dec', 'jul', 'jun', 'mar', 'may', 'nov','oct', 'fri', 'mon', 'thu', 'tue', 'wed', 'Alot_of_Times', 'Few_times','Many_times', 'one_week_till_last_contact','two_weeks_till_last_contact', 'zero_times_contacted','one_time_contacted', 'two_times_contacted', 'three_times_contacted','four_times_contacted', 'previous_campaign_failure', 'Not_campaigned','previous_campaign_successful', 'Fact']]
Y= scaled['target']
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=1,test_size=0.2)

def upSample(X_train, y_train):
    df_all = pd.concat((X_train, pd.DataFrame({'value': y_train}, index=y_train.index)), axis=1)
    
    df_majority = df_all [df_all.value==0]
    df_minority = df_all[df_all.value==1]
     
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=df_majority.shape[0],    # to match majority class
                                     random_state=123) # reproducible results
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled], axis=0)
    y_upsampled = df_upsampled.value
    X_upsampled = df_upsampled.drop('value', axis=1)

    return X_upsampled, y_upsampled
X_train,y_train=upSample(X_train,y_train)
model=LogisticRegression()
model.fit(X_train,y_train)
lrpred=model.predict(X_test)
lrpredprob=model.predict_proba(X_test)
print (classification_report(y_test, lrpred))
cnf_matrix = confusion_matrix(y_test, lrpred, labels=[0,1])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Not_Subscribed(0)','Subscribed(1)'],normalize= False,  title='Confusion matrix')
print(jaccard_score(y_test, lrpred))
print(log_loss(y_test, lrpredprob))
svclass = LinearSVC(random_state=0, tol=1e-5)
svclass.fit(X_train,y_train)
svpred=svclass.predict(X_test)
svclass.score(X_test,y_test)
print (classification_report(y_test, svpred))
cnf_matrix = confusion_matrix(y_test, svpred, labels=[0,1])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Not_Subscribed(0)','Subscribed(1)'],normalize= False,  title='Confusion matrix')
print('Jaccard Similarity Score:',jaccard_score(y_test, svpred))
print('MAE:',mean_absolute_error(svclass.predict(X_test), y_test))
print('MSE:',mean_squared_error(svclass.predict(X_test), y_test))
print('RMSE:',np.sqrt(mean_squared_error(svclass.predict(X_test), y_test)))
rfclassifier=RandomForestClassifier(n_estimators=30,random_state=0,max_depth=27)
rfclassifier.fit(X_train,y_train)
rfpred=rfclassifier.predict(X_test)
rfclassifier.score(X_test,y_test)
print (classification_report(y_test, rfpred))
cnf_matrix = confusion_matrix(y_test, rfpred, labels=[0,1])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Not_Subscribed(0)','Subscribed(1)'],normalize= False,  title='Confusion matrix')
print('Jaccard Similarity Score:',jaccard_score(y_test, rfpred))
print('MAE:',mean_absolute_error(rfclassifier.predict(X_test), y_test))
print('MSE:',mean_squared_error(rfclassifier.predict(X_test), y_test))
print('RMSE:',np.sqrt(mean_squared_error(rfclassifier.predict(X_test), y_test)))
#parameters={
#'n_estimators':[20,30,40],
#'max_depth':range(20,30),
#'criterion' :['entropy']
#}
#rfcv = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv= 5,n_jobs=-1)
#rfcv.fit(X_train,y_train)
#rfcv.best_params_
#rfcv.best_estimator_.fit(X_train,y_train)
#rfcv1pred=rfcv.best_estimator_.predict(X_test)
#rfcv.best_estimator_.score(X_test,y_test)
#print (classification_report(y_test, rfcv1pred))
#cnf_matrix = confusion_matrix(y_test, rfcv1pred, labels=[0,1])
#np.set_printoptions(precision=2)
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=['Not_Subscribed(0)','Subscribed(1)'],normalize= False,  title='Confusion matrix')
#from sklearn import tree
#tree.plot_tree(rfcv.best_estimator_.estimators_[39])
knnclassifier=KNeighborsClassifier(n_neighbors=5)
knnclassifier.fit(X_train,y_train)
knnpred=knnclassifier.predict(X_test)

knnclassifier.score(X_test,y_test)
print (classification_report(y_test, knnpred))
cnf_matrix = confusion_matrix(y_test, knnpred, labels=[0,1])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Not_Subscribed(0)','Subscribed(1)'],normalize= False,  title='Confusion matrix')
print('Jaccard Similarity Score:',jaccard_score(y_test, knnpred))
print('MAE:',mean_absolute_error(knnclassifier.predict(X_test), y_test))
print('MSE:',mean_squared_error(knnclassifier.predict(X_test), y_test))
print('RMSE:',np.sqrt(mean_squared_error(knnclassifier.predict(X_test), y_test)))
#parameters={
#'leaf_size':[30,40],
#'n_neighbors':range(5,15)
#}
#knncv = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parameters, cv= 5,n_jobs=-1)
#knncv.fit(X_train,y_train)
#knncv.best_params_
#knncv.best_estimator_.fit(X_train,y_train)
#knncv1pred=knncv.best_estimator_.predict(X_test)
#knncv.best_estimator_.score(X_test,y_test)
#print (classification_report(y_test, knncv1pred))
#cnf_matrix = confusion_matrix(y_test, knncv1pred, labels=[0,1])
#np.set_printoptions(precision=2)
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=['Not_Subscribed(0)','Subscribed(1)'],normalize= False,  title='Confusion matrix')
tree= DecisionTreeClassifier(criterion="entropy",random_state=1,max_depth=8)
tree.fit(X_train,y_train)
treepred=tree.predict(X_test)

tree.score(X_test,y_test)
print (classification_report(y_test, treepred))
cnf_matrix = confusion_matrix(y_test, treepred, labels=[0,1])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Not_Subscribed(0)','Subscribed(1)'],normalize= False,  title='Confusion matrix')
print('Jaccard Similarity Score:',jaccard_score(y_test, treepred))
print('MAE:',mean_absolute_error(tree.predict(X_test), y_test))
print('MSE:',mean_squared_error(tree.predict(X_test), y_test))
print('RMSE:',np.sqrt(mean_squared_error(tree.predict(X_test), y_test)))
#parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,15),'criterion':["entropy",'gini']}
#treecv = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parameters, cv= 5,n_jobs=-1)
#treecv.fit(X_train,y_train)
#treecv.best_params_
#treecv.best_estimator_.fit(X_train,y_train)
#treecv1pred=treecv.best_estimator_.predict(X_test)
#treecv.best_estimator_.score(X_test,y_test)
#print (classification_report(y_test, treecv1pred))
#nf_matrix = confusion_matrix(y_test, treecv1pred, labels=[0,1])
#np.set_printoptions(precision=2)
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=['Not_Subscribed(0)','Subscribed(1)'],normalize= False,  title='Confusion matrix')
