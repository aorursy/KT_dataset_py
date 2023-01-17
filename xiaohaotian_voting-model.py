import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

TRAIN_PATH=r"../input/train.csv"
TEST_PATH=r"../input/test.csv"

train_df=pd.read_csv(TRAIN_PATH)
test_df=pd.read_csv(TEST_PATH)

reduced_flag=0#whether to perform dimension reduction
dummy_flag=0#whether to perform dummy transformation
#import model
#ensemble model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
#linear model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
#knn
from sklearn.neighbors import KNeighborsClassifier
#navies bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
#tree model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
#svm
from sklearn.svm import SVC
#xgb
from xgboost import XGBClassifier
#lgb
import lightgbm as lgb
#model selection
from sklearn.feature_selection import SelectFromModel
#metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#model selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
#combine data, put train and test data together
def combine_data():
    target=train_df['Survived']
    combined=pd.concat([train_df.drop(['Survived'],axis=1),test_df])
    passenger_id=combined[['PassengerId']]
    combined.drop(['PassengerId'],axis=1,inplace=True)
    combined.reset_index(inplace=True)
    passenger_id.reset_index(inplace=True)
    return combined,target,passenger_id
combined,target,passenger_id=combine_data()
#name feature
def process_name(df):
    df['Title']=df.Name.str.extract('([A-Za-z]+)\.')
    #df['Name_length']=df['Name'].map(len)
    age_dict={
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty",
    "Dona" : "Royalty"
    }
    df['Title']=df['Title'].map(age_dict)
process_name(combined)
#age feature
train=combined[:891]
test=combined[891:]
train_age_group=train.groupby(['Sex','Title','Pclass']).mean()
train_age_group=train_age_group.reset_index()[['Sex','Title','Pclass','Age']]
def fill_age(row):
    age_missing=train_age_group.loc[(train_age_group['Sex']==row['Sex'])&
                                   (train_age_group['Title']==row['Title'])&
                                   (train_age_group['Pclass']==row['Pclass']),'Age'].values[0]
    return age_missing
                                    
def process_age(df):
    #fill in missing ages with respect to the Sex, Pclass, title feature, we use train data to form the age dict
    #predict missing age
    train_age_df=train[['Age','Pclass','Sex','Title']]
    all_age_df=df[['Age','Pclass','Sex','Title']]
    train_age_df=pd.get_dummies(train_age_df)
    all_age_df=pd.get_dummies(all_age_df)
    known_age=train_age_df.loc[train_age_df.Age.notnull()]
    unknown_age=all_age_df.loc[all_age_df.Age.isnull()]
    Y=known_age[['Age']]
    X=known_age.drop(['Age'],axis=1)
    rfr=RandomForestRegressor(random_state=0,n_estimators=100)
    rfr.fit(X,Y)
    pre_age=rfr.predict(unknown_age.drop(['Age'],axis=1))
    df.loc[df.Age.isnull(),'Age']=pre_age
    #df['Age']=df.apply(lambda row:fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
process_age(combined)
#family size feature
def process_family_size(df):
    df['Family_size']=df['SibSp']+df['Parch']+1
    df['Singleton']=0
    df['Small_family']=0
    df['Big_family']=0
    df.loc[df['Family_size']==1,'Singleton']=1
    df.loc[(df['Family_size']>=2)&(df['Family_size']<5),'Small_family']=1
    df.loc[df['Family_size']>=5,'Big_family']=1
process_family_size(combined)
#cabin feature
def process_cabin(df):
    df['Cabin_cap']=df.Cabin.str.extract('([A-Za-z])')
    df['Cabin_cap'].fillna('U',inplace=True)
process_cabin(combined)
#sex feature
def process_sex(df):
    df['Sex']=df['Sex'].map({'male':1,'female':0})
process_sex(combined)
#embarked feature
def process_embarked(df):
    df['Embarked'].fillna('S',inplace=True)
process_embarked(combined)
#ticket feature
def process_ticket(df):
#     # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
#     def cleanTicket(ticket):
#         ticket = ticket.replace('.','')
#         ticket = ticket.replace('/','')
#         ticket = ticket.split()
#         ticket = map(lambda t : t.strip(), ticket)
#         ticket = list(filter(lambda t : not t.isdigit(), ticket))
#         if len(ticket) > 0:
#             return ticket[0]
#         else: 
#             return 'XXX'
#     # Extracting dummy variables from tickets:
#     df['Ticket'] = df['Ticket'].map(cleanTicket)
    def Ticket_count_group(num):
        if (num>=2)&(num<=4):
            return 0
        elif(num==1)|((num>=5)&(num<=8)):
            return 1
        else:
            return 2
    ticket_dict=dict(df.groupby(['Ticket'])['index'].count())
    df['TicketCount']=df['Ticket'].map(ticket_dict)
    df['TicketCount']=df['TicketCount'].map(Ticket_count_group)
#     df['TicketCount']=2
#     df['TicketCount']=df.apply(lambda row: 0 if(ticket_dict[row['Ticket']] > 2) & (ticket_dict[row['Ticket']] < 4) else 0,axis=1)
#     df['TicketCount']=df.apply(lambda row: 1 if (ticket_dict[row['Ticket']] == 1) | (ticket_dict[row['Ticket']] <=8 & ticket_dict[row['Ticket']]>=5) else 0,axis=1)
process_ticket(combined)
#fare feature
def process_fare(df):
    df['Fare'].fillna(df['Fare'].median(),inplace=True)
process_fare(combined)
#surname feature
def process_surname(df):
    df['Surname']=df['Name'].map(lambda x:x.split(',')[0])
process_surname(combined)
#did one family member survive and did all family members die
def process_family_survival(df):
    df['Family_one_survived']=0
    df['Family_all_died']=0
    train=train_df
    process_surname(train)
    for idx in range(len(df['Name'])):
        passengerId=passenger_id.loc[idx]['PassengerId']
        surname=df.loc[idx]['Surname']
        family_survival=train.loc[((train['Surname']==surname)&(train['PassengerId']!=passengerId)),'Survived'].mean()
        if(not np.isnan(family_survival) and family_survival > 0.0):
            df.loc[idx,'Family_one_survived']=1
        if(not np.isnan(family_survival) and family_survival == 0.0):
            df.loc[idx,'Family_all_died']=1
process_family_survival(combined)          
#feature preprocess
def fill_family_type(row):
    if row['Singleton']==1:
        return 0
    elif row['Big_family']==1:
        return 1
    else:
        return 2

def process_features_dummy(df_raw):
    df=df_raw
    pclass_dummy=pd.get_dummies(data=df,columns=['Pclass'],prefix='Pclass')
    df[pclass_dummy.columns]=pclass_dummy
    title_dummy=pd.get_dummies(data=df,columns=['Title'],prefix='Title')
    df[title_dummy.columns]=title_dummy
    #cabin_dummy=pd.get_dummies(data=df,columns=['Cabin_cap'],prefix='Cabin_cap')
   # df[cabin_dummy.columns]=cabin_dummy
    embark_dummy=pd.get_dummies(data=df,columns=['Embarked'],prefix='Embarked')
    df[embark_dummy.columns]=embark_dummy
    sex_dummy=pd.get_dummies(data=df,columns=['Sex'],prefix='Sex')
    df[sex_dummy.columns]=sex_dummy
   # ticket_dummy=pd.get_dummies(data=df,columns=['TicketCount'],prefix='TicketCount')
   # df[ticket_dummy.columns]=ticket_dummy
    return df.drop(['Name','Ticket','Cabin','Embarked','Title','index','Pclass','Sex','TicketCount','Cabin_cap'],axis=1)
    
def process_features(df_raw):
    df=df_raw
    df['Title']=df['Title'].map({'Mr':0,'Officer':1,'Master':2,'Royalty':3,'Miss':4,'Mrs':5})
    df['Cabin_cap']=df['Cabin_cap'].map({'T':0,'U':1,'A':2,'G':3,'C':4,'F':5,'E':6,'D':7,'B':8})
    df['Embarked']=df['Embarked'].map({'S':0,'Q':1,'C':2})
    df['FamilyType']=df.apply(lambda row :fill_family_type(row), axis=1)
    return df.drop(['Name','Ticket','Cabin','index','Singleton','Small_family','Big_family','Surname'],axis=1)

combine_raw=process_features(combined)
combine_dummy=process_features_dummy(combined)
if dummy_flag==1:
    combined=combine_dummy
else:
    combined=combine_raw
#generate train and test data
def get_train_test_data(df):
    return df[:891],df[891:],target
train,test,target=get_train_test_data(combined)
train_x,vali_x,train_y,vali_y=train_test_split(train,target,test_size=0.3,random_state=0)
#model compare
def compare_model(clf_list,train_x,train_y,cv):
    mla_columns=['model_name','train_accuracy_mean','test_accuracy_mean','mla Test Accuracy 3*STD','fit_time']
    mla_pd=pd.DataFrame(columns=mla_columns)
    for idx,clf in enumerate(clf_list):
        mla_pd.loc[idx,'model_name']=clf.__class__.__name__
        cv_result=cross_validate(clf,X=train_x,y=train_y,cv=cv)
        mla_pd.loc[idx,'train_accuracy_mean']=cv_result['train_score'].mean()
        mla_pd.loc[idx,'test_accuracy_mean']=cv_result['test_score'].mean()
        mla_pd.loc[idx, 'mla Test Accuracy 3*STD']=cv_result['test_score'].std()*3  
        mla_pd.loc[idx,'fit_time']=cv_result['fit_time'].mean()
    mla_pd.sort_values(by='test_accuracy_mean',ascending=False,inplace=True)
    return mla_pd

#tune model
def tune_model(clf,param_grid,train_x,train_y,vali_x,vali_y,cv):
    print('----',clf.__class__.__name__,'fine tune','----')
    grid_search=GridSearchCV(clf,param_grid=param_grid,cv=cv,scoring='accuracy',verbose=True)
    grid_search.fit(train_x,train_y)
    vali_score=accuracy_score(vali_y,grid_search.predict(vali_x))
    print('best score :{} best param:{} vali score:{}'.format(grid_search.best_score_, grid_search.best_params_, vali_score))
    print('----',clf.__class__.__name__,'fine tune finish','----')
    return grid_search.best_params_

#feature dimension reduce
def reduce_dimension():
    pass
clf_lst=[
    #Ensemble Methods
    AdaBoostClassifier(),
    BaggingClassifier(),
    ExtraTreesClassifier(),
    GradientBoostingClassifier(),
    RandomForestClassifier(),
    #GLM
    LogisticRegression(),
    PassiveAggressiveClassifier(),
    RidgeClassifierCV(),
    SGDClassifier(),
    Perceptron(),
    #navies bayes
    BernoulliNB(),
    GaussianNB(),
    #SVM
    SVC(probability=True),
    #Trees    
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    #xgb
    XGBClassifier()
]
mla_pd=compare_model(clf_lst,train,target,5)
mla_pd.reset_index(inplace=True)
mla_pd
#choose models of which test_accuracy_mean is over 0.8 as base model of voting model
#xgb,gbc,ada,rf,lr,bag
grid_learn=[0.01, 0.03, 0.05, 0.1, 0.25,0.5]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_n_estimator = [50,100,200]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]
cv=10

xgb_param_grid={
    'learning_rate':grid_learn,
    'max_depth':[1,2,4,6,8,10],
    'n_estimators':grid_n_estimator,
    'seed':grid_seed
}

gbc_param_grid={
    'learning_rate': [0.05],
    'n_estimators': [300],
    'max_depth': grid_max_depth,
    'random_state': grid_seed
}

ada_param_grid={
    'n_estimators': grid_n_estimator,
    'learning_rate': grid_learn, 
    'random_state': grid_seed
}

rf_param_grid={
    'n_jobs':[-1],
    'n_estimators': grid_n_estimator,
    'criterion': grid_criterion,
    'max_depth': grid_max_depth,
    'oob_score': [True],
    'random_state': grid_seed
}

lr_param_grid={
    'fit_intercept': grid_bool,
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'random_state': grid_seed
}

bag_param_grid={
    'n_estimators': grid_n_estimator,
    'max_samples': grid_ratio,
    'random_state': grid_seed
}

svm_param_grid={
    'kernel':['linear'],
    'C':[0.025, 0.5, 1]
}
#xgb,gbc,ada,rf,lr,bag
xgb_best_param=tune_model(XGBClassifier(),xgb_param_grid,train_x,train_y,vali_x,vali_y,cv)
ada_best_param=tune_model(AdaBoostClassifier(),ada_param_grid,train_x,train_y,vali_x,vali_y,cv)
gbc_best_param=tune_model(GradientBoostingClassifier(),gbc_param_grid,train_x,train_y,vali_x,vali_y,cv)
rf_best_param=tune_model(RandomForestClassifier(),rf_param_grid,train_x,train_y,vali_x,vali_y,cv)
lr_best_param=tune_model(LogisticRegression(),lr_param_grid,train_x,train_y,vali_x,vali_y,cv)
bag_best_param=tune_model(BaggingClassifier(),bag_param_grid,train_x,train_y,vali_x,vali_y,cv)
svm_best_param=tune_model(SVC(probability=True),svm_param_grid,train_x,train_y,vali_x,vali_y,cv)
#voting model
#xgb,gbc,ada,rf,lr,bag,svm
estimators=[
    ('xgb',XGBClassifier(**xgb_best_param)),
    ('gbc',GradientBoostingClassifier(**gbc_best_param)),
    ('ada',AdaBoostClassifier(**ada_best_param)),
    ('rf',RandomForestClassifier(**rf_best_param)),
    ('lr',LogisticRegression(**lr_best_param)),
    ('bag',BaggingClassifier(**bag_best_param)),
    ('svm',SVC(**svm_best_param,probability=True))
]
voting_model=VotingClassifier(estimators=estimators,voting='soft')
voting_model.fit(train_x,train_y)
y_pre=voting_model.predict(vali_x)
accuracy_score(y_pre,vali_y)