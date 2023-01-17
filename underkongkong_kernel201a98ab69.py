
import sys
print(sys.argv[0])
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')
PassengerId=test['PassengerId']
all_data = pd.concat([train, test], ignore_index = True)
train.head()
train.info()
sns.barplot(x="Pclass", y="Survived", data=train);

sns.barplot(x="Sex", y="Survived", data=train);
sns.barplot(x="SibSp", y="Survived", data=train);
sns.barplot(x="Parch", y="Survived", data=train);
facet = sns.FacetGrid(train, hue="Survived",aspect=2)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlabel('Age') 
plt.ylabel('density') 
sns.countplot('Embarked',hue='Survived',data=train)
all_data['Title'] = all_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))

all_data['Title'] = all_data['Title'].map(Title_Dict)
sns.barplot(x="Title", y="Survived", data=all_data)
all_data['FamilySize']=all_data['SibSp']+all_data['Parch']+1
sns.barplot(x="FamilySize", y="Survived", data=all_data)
def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0
all_data['FamilyLabel']=all_data['FamilySize'].apply(Fam_label)
sns.barplot(x="FamilyLabel", y="Survived", data=all_data)
all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
all_data['Deck']=all_data['Cabin'].str.get(0)
sns.barplot(x="Deck", y="Survived", data=all_data)
Ticket_Count = dict(all_data['Ticket'].value_counts())
all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x:Ticket_Count[x])
sns.barplot(x='TicketGroup', y='Survived', data=all_data)
def Ticket_Label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0

all_data['TicketGroup'] = all_data['TicketGroup'].apply(Ticket_Label)
sns.barplot(x='TicketGroup', y='Survived', data=all_data)
from sklearn.ensemble import RandomForestRegressor
age_df = all_data[['Age', 'Pclass','Sex','Title']]
age_df=pd.get_dummies(age_df)
known_age = age_df[age_df.Age.notnull()].values
unknown_age = age_df[age_df.Age.isnull()].values
y = known_age[:, 0]
X = known_age[:, 1:]
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(X, y)
predictedAges = rfr.predict(unknown_age[:, 1::])
all_data.loc[ (all_data.Age.isnull()), 'Age' ] = predictedAges 
all_data[all_data['Embarked'].isnull()]
all_data.groupby(by=["Pclass","Embarked"]).Fare.median()
all_data['Embarked'] = all_data['Embarked'].fillna('C')
fare=all_data[(all_data['Embarked'] == "S") & (all_data['Pclass'] == 3)].Fare.median()
all_data['Fare']=all_data['Fare'].fillna(fare)
all_data[all_data['Fare'].isnull()]
all_data['Surname']=all_data['Name'].apply(lambda x:x.split(',')[0].strip())
Surname_Count = dict(all_data['Surname'].value_counts())
all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x:Surname_Count[x])
Female_Child_Group=all_data.loc[(all_data['FamilyGroup']>=2) & ((all_data['Age']<=12) | (all_data['Sex']=='female'))]
Male_Adult_Group=all_data.loc[(all_data['FamilyGroup']>=2) & (all_data['Age']>12) & (all_data['Sex']=='male')]
Female_Child=pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
Female_Child.columns=['GroupCount']
Female_Child
sns.barplot(x=Female_Child.index, y=Female_Child["GroupCount"]).set_xlabel('AverageSurvived');
Male_Adult=pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
Male_Adult.columns=['GroupCount']
Male_Adult
sns.barplot(x=Male_Adult.index, y=Female_Child["GroupCount"]).set_xlabel('AverageSurvived');
Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
print(Dead_List)
Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)
print(Survived_List)
train=all_data.loc[all_data['Survived'].notnull()]
test=all_data.loc[all_data['Survived'].isnull()]
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 'male'
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 'Mr'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 'female'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'
all_data=pd.concat([train, test])
all_data=all_data[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilyLabel','Deck','TicketGroup']]
all_data=pd.get_dummies(all_data)
train=all_data[all_data['Survived'].notnull()]
test=all_data[all_data['Survived'].isnull()].drop('Survived',axis=1)
X = train.values[:,1:]
y = train.values[:,0]
all_data.info()
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest

pipe=Pipeline([('select',SelectKBest(k=20)), 
               ('classify', RandomForestClassifier(random_state = 10, n_estimators=100,max_features = 'sqrt'))])

param_test = {
            'classify__n_estimators':list(range(20,50,2)), 
              'classify__max_depth':list(range(3,10,3))}
             
gsearch = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='roc_auc', cv=10,verbose=True)
gsearch.fit(X,y)
print(gsearch.best_params_, gsearch.best_score_)
from sklearn.pipeline import make_pipeline
select = SelectKBest(k = 20)
clf = RandomForestClassifier(random_state = 10, warm_start = True, 
                                  n_estimators = 26,
                                  max_depth = 6, 
                                  max_features = 'sqrt')
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)

from sklearn import model_selection, metrics
cv_score = model_selection.cross_val_score(pipeline, X, y, cv= 10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))
print(cv_score)
from sklearn.metrics import roc_curve,auc,precision_recall_curve
from sklearn.model_selection import KFold,StratifiedKFold

classifier=pipeline
n_samples,n_features = X.shape
random_state = np.random.RandomState(0)

kfold = StratifiedKFold(n_splits=5) # 使用5折交叉验证，并且画PR曲线
cv = kfold.split(X,y)


for i,(train_num,test_num) in enumerate(cv): # 
    probas_ = classifier.fit(X[train_num],y[train_num]).predict_proba(X[test_num])
    precision,recall,thresholds= precision_recall_curve(y[test_num],probas_[:,1])# 通过precision_recall_curve()函数，求出recall，precision，以及阈值
    plt.plot(recall,precision,lw=1)


    
plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6),label="Luck") # 画对角线
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel("Recall Rate")
plt.ylabel("Precision Rate")
plt.show()




from sklearn.metrics import roc_curve,auc,precision_recall_curve
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import confusion_matrix

classifier=pipeline
n_samples,n_features = X.shape
random_state = np.random.RandomState(0)

kfold = StratifiedKFold(n_splits=5) # 使用5折交叉验证，并且画ROC曲线
cv = kfold.split(X,y)

for i,(train_num,test_num) in enumerate(cv): # 通过训练数据，使用svm线性核建立模型，并对测试集进行测试，求出预测得分
    probas_ = classifier.fit(X[train_num],y[train_num]).predict_proba(X[test_num])
    fpr, tpr, thresholds=roc_curve(y[test_num],probas_[:,1])
    plt.plot(fpr,tpr,lw=1)
    roc_auc = auc(fpr,tpr) 
    print('AUC_',i,': ',roc_auc)



plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print('\n')
print('模型混淆矩阵为\n',confusion_matrix(y[test_num],np.round(probas_[:,1])))
predictions = pipeline.predict(test)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv("/kaggle/working/submission_t6.csv", index=False)
sel=SelectKBest(k=20)
X=sel.fit_transform(X,y)
test=sel.transform(test)
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
model=xgb.XGBClassifier(random_state=10)
param_grid={
            'learning_rate':np.arange(0.02,0.2,0.03)
           ,'max_depth':np.arange(1,10,2)
           ,'max_features':np.arange(1,6,2)
            ,'subsample':np.arange(0,1,0.2)}
model = GridSearchCV(model,param_grid = param_grid, cv=10, 
                                     scoring="roc_auc", n_jobs= -1, verbose = 1)
model.fit(X,y)
print(model.best_params_, model.best_score_)
from sklearn.pipeline import make_pipeline
select = SelectKBest(k = 20)
clf =xgb.XGBClassifier(random_state = 10,
                    learning_rate=0.08,
                    subsample=1,
                    max_depth = 3, 
                    max_features = 1)
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)

from sklearn import model_selection, metrics
cv_score = model_selection.cross_val_score(pipeline, X, y, cv= 10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))
cmodel=xgb.XGBClassifier(learning_rate=0.08,max_depth=3,max_features=1,subsample=1)
cparam_grid={
            'colsample_bytree':np.arange(0.01,0.99,0.15)
            ,'colsample_bylevel':np.arange(0.01,0.99,0.15)}
cmodel = GridSearchCV(cmodel,param_grid = cparam_grid, cv=10, 
                                     scoring="roc_auc", n_jobs= -1, verbose = 1)
cmodel.fit(X,y)
print(cmodel.best_params_, cmodel.best_score_)
from sklearn.pipeline import make_pipeline
select = SelectKBest(k = 20)
clf =xgb.XGBClassifier(n_estimator=40,
    random_state = 10,
                    learning_rate=0.08,
                    subsample=0.9,
                    max_depth = 3, 
                    max_features = 1,
                     colsample_bylevel=0.46,
                      colsample_bytree=0.91)
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)
from sklearn import model_selection, metrics
cv_score = model_selection.cross_val_score(pipeline, X, y, cv= 10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))

classifier=pipeline
n_samples,n_features = X.shape
random_state = np.random.RandomState(0)

kfold = StratifiedKFold(n_splits=5) # 使用5折交叉验证，并且画PR曲线
cv = kfold.split(X,y)


for i,(train_num,test_num) in enumerate(cv): # 
    probas_ = classifier.fit(X[train_num],y[train_num]).predict_proba(X[test_num])
    precision,recall,thresholds= precision_recall_curve(y[test_num],probas_[:,1])# 通过precision_recall_curve()函数，求出recall，precision，以及阈值
    plt.plot(recall,precision,lw=1)
    


    
plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6),label="Luck") # 画对角线
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel("Recall Rate")
plt.ylabel("Precision Rate")
plt.show()




classifier=pipeline
n_samples,n_features = X.shape
random_state = np.random.RandomState(0)

kfold = StratifiedKFold(n_splits=5) # 使用5折交叉验证，并且画ROC曲线
cv = kfold.split(X,y)

for i,(train_num,test_num) in enumerate(cv): # 通过训练数据，使用svm线性核建立模型，并对测试集进行测试，求出预测得分
    probas_ = classifier.fit(X[train_num],y[train_num]).predict_proba(X[test_num])
    fpr, tpr, thresholds=roc_curve(y[test_num],probas_[:,1])
    plt.plot(fpr,tpr,lw=1)
    roc_auc = auc(fpr,tpr) 
    print('AUC_',i,': ',roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
print('\n')
print('模型混淆矩阵为\n',confusion_matrix(y[test_num],np.round(probas_[:,1])))
#TitanicLRmodel
model_y=cmodel.predict(test)
modelrf_y=model_y.astype(int)
#导出预测结果
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": modelrf_y})
#将预测结果导出为csv文件
submission.to_csv('/kaggle/working/submission_t4.csv',index=False)
from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(random_state=0)
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }
modelgsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=10, 
                                     scoring="roc_auc", n_jobs= -1, verbose = 1)
modelgsGBC.fit(X,y)
print(modelgsGBC.best_params_, modelgsGBC.best_score_)
from sklearn.pipeline import make_pipeline
select = SelectKBest(k = 20)
clf =GradientBoostingClassifier(n_estimators=300,
    random_state = 0,
                    learning_rate=0.1,
                  loss='deviance',
                    max_depth = 4, 
                    max_features = 0.3,
                    min_samples_leaf=100,
                     )
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)
from sklearn import model_selection, metrics
cv_score = model_selection.cross_val_score(pipeline, X, y, cv= 10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))

classifier=pipeline
n_samples,n_features = X.shape
random_state = np.random.RandomState(0)

kfold = StratifiedKFold(n_splits=5) # 使用5折交叉验证，并且画PR曲线
cv = kfold.split(X,y)


for i,(train_num,test_num) in enumerate(cv): # 
    probas_ = classifier.fit(X[train_num],y[train_num]).predict_proba(X[test_num])
    precision,recall,thresholds= precision_recall_curve(y[test_num],probas_[:,1])# 通过precision_recall_curve()函数，求出recall，precision，以及阈值
    plt.plot(recall,precision,lw=1)


    
plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6),label="Luck") # 画对角线
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel("Recall Rate")
plt.ylabel("Precision Rate")
plt.show()
from sklearn.metrics import roc_curve,auc,precision_recall_curve
from sklearn.model_selection import KFold,StratifiedKFold

classifier=pipeline
n_samples,n_features = X.shape
random_state = np.random.RandomState(0)

kfold = StratifiedKFold(n_splits=5) # 使用5折交叉验证，并且画ROC曲线
cv = kfold.split(X,y)

for i,(train_num,test_num) in enumerate(cv): # 通过训练数据，使用svm线性核建立模型，并对测试集进行测试，求出预测得分
    probas_ = classifier.fit(X[train_num],y[train_num]).predict_proba(X[test_num])
    fpr, tpr, thresholds=roc_curve(y[test_num],probas_[:,1])
    plt.plot(fpr,tpr,lw=1)
    roc_auc = auc(fpr,tpr) 
    print('AUC_',i,': ',roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print('\n')
print('模型混淆矩阵为\n',confusion_matrix(y[test_num],np.round(probas_[:,1])))
#TitanicLRmodel
model_y=modelgsGBC.predict(test)
modelrf_y=model_y.astype(int)
#导出预测结果
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": modelrf_y})
#将预测结果导出为csv文件
submission.to_csv('/kaggle/working/submission_t5.csv',index=False)
GBC = GradientBoostingClassifier(random_state=0)
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : np.arange(10,200,30),
              'learning_rate': np.arange(0.05,0.15,0.04),
              'max_depth': np.arange(5,10),
              
              }
zmodel = GridSearchCV(GBC,param_grid = gb_param_grid, cv=10, 
                                     scoring="roc_auc", n_jobs= -1, verbose = 1)
zmodel.fit(X,y)
print(zmodel.best_params_, zmodel.best_score_)
#GradientBoostingClassifier模型
GBC = GradientBoostingClassifier(n_estimators=40,learning_rate=0.05,max_depth=5)
gb_param_grid = {'loss' : ["deviance"],
              'min_samples_leaf':np.arange(80,120,5)
              ,'max_features': np.arange(0.05,0.15,0.01)
              }
zzmodel = GridSearchCV(GBC,param_grid = gb_param_grid, cv=10, 
                                     scoring="roc_auc", n_jobs= -1, verbose = 1)
zzmodel.fit(X,y)
print(zzmodel.best_params_, zzmodel.best_score_)

classifier=pipeline
n_samples,n_features = X.shape
random_state = np.random.RandomState(0)

kfold = StratifiedKFold(n_splits=5) # 使用5折交叉验证，并且画pr曲线
cv = kfold.split(X,y)


for i,(train_num,test_num) in enumerate(cv): # 
    probas_ = classifier.fit(X[train_num],y[train_num]).predict_proba(X[test_num])
    precision,recall,thresholds= precision_recall_curve(y[test_num],probas_[:,1])# 通过precision_recall_curve()函数，求出recall，precision，以及阈值
    plt.plot(recall,precision,lw=1)


    
plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6),label="Luck") # 画对角线
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel("Recall Rate")
plt.ylabel("Precision Rate")
plt.show()
classifier=pipeline
n_samples,n_features = X.shape
random_state = np.random.RandomState(0)

kfold = StratifiedKFold(n_splits=5) # 使用5折交叉验证，并且画ROC曲线
cv = kfold.split(X,y)

for i,(train_num,test_num) in enumerate(cv): # 通过训练数据，使用svm线性核建立模型，并对测试集进行测试，求出预测得分
    probas_ = classifier.fit(X[train_num],y[train_num]).predict_proba(X[test_num])
    fpr, tpr, thresholds=roc_curve(y[test_num],probas_[:,1])
    plt.plot(fpr,tpr,lw=1)
    roc_auc = auc(fpr,tpr) 
    print('AUC_',i,': ',roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
print('\n')
print('模型混淆矩阵为\n',confusion_matrix(y[test_num],np.round(probas_[:,1])))
from sklearn.pipeline import make_pipeline
select = SelectKBest(k = 20)
clf =GradientBoostingClassifier(n_estimators=40,
    random_state = 0,
                    learning_rate=0.05,
                  loss='deviance',
                    max_depth = 5, 
                     )
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)
from sklearn import model_selection, metrics
cv_score = model_selection.cross_val_score(pipeline, X, y, cv= 10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))