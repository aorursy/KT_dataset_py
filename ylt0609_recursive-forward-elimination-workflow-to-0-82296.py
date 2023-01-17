# loading package
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline
sns.set()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix, classification_report
# loading data
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_data = df_train.append(df_test)
# for display dataframe
from IPython.display import display
from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
# ignore warning
import warnings
warnings.filterwarnings("ignore")
# Convert Sex
df_data['Sex_Code'] = df_data['Sex'].map({'female' : 1, 'male' : 0}).astype('int')
# split training set the testing set
df_train = df_data[:len(df_train)]
df_test = df_data[len(df_train):]
# Inputs set and labels
X = df_train.drop(labels=['Survived','PassengerId'],axis=1)
Y = df_train['Survived']
# Show Baseline
Base = ['Sex_Code','Pclass']
Base_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
Base_Model.fit(X[Base], Y)
print('Base oob score :%.5f' %(Base_Model.oob_score_),'   LB_Public : 0.76555')
# submission if you want
'''# submits
X_Submit = df_test.drop(labels=['PassengerId'],axis=1)

Base_pred = Base_Model.predict(X_Submit[Base])

submit = pd.DataFrame({"PassengerId": df_test['PassengerId'],
                      "Survived":Base_pred.astype(int)})
submit.to_csv("submit_Base.csv",index=False)''';
# A function about in-sample correct dataframe
def Correct_classified_df(model, training_df, labels_df):
    kfold = StratifiedKFold(n_splits=10)
    correct_X = training_df
    corret_classified_index = []
    # fit in-sample by cross- validation
    for train_index, val_index in kfold.split(correct_X, labels_df):
        #print("Train:",train_index,"Val:",val_index);
        FITT = model.fit(X = correct_X.iloc[train_index], y = labels_df.iloc[train_index])
        pred = FITT.predict(correct_X.iloc[val_index,:])
        #print(pred)
        #print(labels_df.iloc[val_index])
        corret_classified_index.append(
        labels_df.iloc[val_index][pred == labels_df.iloc[val_index]].index.values)
        #print(correct_X.iloc[val_index])
    whole_index=np.concatenate(corret_classified_index)

    # whole_index
    correct_classified_df = correct_X[correct_X.index.isin(whole_index)]
    correct_classified_df['Survived'] = labels_df[correct_X.index.isin(whole_index)]
    return correct_classified_df
# In-sample correct
Base_correct = Correct_classified_df(Base_Model, X[Base], Y)
# Compare with what we wish
tem = [df_data, Base_correct]
tem_factor_df = []
for i in tem:
    tem_factor_df.append(pd.pivot_table( i, values='Survived',index='Sex_Code',columns='Pclass').round(3))
# display
display(pd.concat([tem_factor_df[0], tem_factor_df[1]],keys=['Data', 'Base'], axis = 1))
# visualization
g = sns.FacetGrid(data=df_data, row='Sex', col='Pclass',margin_titles=True)
g.map(sns.countplot,'Survived')
# there is some bugs in log-scale of boxplot. 
# alternatively, we transform x into log10(x) for visualization.
fig, ax = plt.subplots( figsize = (18,7) )
df_data['Log_Fare'] = (df_data['Fare']+1).map(lambda x : np.log10(x) if x > 0 else 0)
sns.boxplot(y='Pclass', x='Log_Fare',hue='Survived',data=df_data, orient='h'
                ,ax=ax,palette="Set3")
ax.set_title(' Log_Fare & Pclass vs Survived ',fontsize = 20)
pd.pivot_table(df_data,values = ['Fare'], index = ['Pclass'], columns= ['Survived'] ,aggfunc = 'median' ).round(3)
# Filling missing values
df_data['Fare'] = df_data['Fare'].fillna(df_data['Fare'].median())

# Making Bins
df_data['FareBin_4'] = pd.qcut(df_data['Fare'], 4)
df_data['FareBin_5'] = pd.qcut(df_data['Fare'], 5)
df_data['FareBin_6'] = pd.qcut(df_data['Fare'], 6)

label = LabelEncoder()
df_data['FareBin_Code_4'] = label.fit_transform(df_data['FareBin_4'])
df_data['FareBin_Code_5'] = label.fit_transform(df_data['FareBin_5'])
df_data['FareBin_Code_6'] = label.fit_transform(df_data['FareBin_6'])

# cross tab
df_4 = pd.crosstab(df_data['FareBin_Code_4'],df_data['Pclass'])
df_5 = pd.crosstab(df_data['FareBin_Code_5'],df_data['Pclass'])
df_6 = pd.crosstab(df_data['FareBin_Code_6'],df_data['Pclass'])

display_side_by_side(df_4,df_5,df_6)

# plots
fig, [ax1, ax2, ax3] = plt.subplots(1, 3,sharey=True)
fig.set_figwidth(18)
for axi in [ax1, ax2, ax3]:
    axi.axhline(0.5,linestyle='dashed', c='black',alpha = .3)
g1 = sns.factorplot(x='FareBin_Code_4', y="Survived", data=df_data,kind='bar',ax=ax1)
g2 = sns.factorplot(x='FareBin_Code_5', y="Survived", data=df_data,kind='bar',ax=ax2)
g3 = sns.factorplot(x='FareBin_Code_6', y="Survived", data=df_data,kind='bar',ax=ax3)
# close FacetGrid object
plt.close(g1.fig)
plt.close(g2.fig)
plt.close(g3.fig)
# splits again beacuse we just engineered new feature
df_train = df_data[:len(df_train)]
df_test = df_data[len(df_train):]
# Training set and labels
X = df_train.drop(labels=['Survived','PassengerId'],axis=1)
Y = df_train['Survived']
# show columns
X.columns
# oob : 0.8054 split = 20,30,40 oob : 0.79012 split = 50
bin_4 = ['Sex_Code','Pclass','FareBin_Code_4']
bin_4_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
bin_4_Model.fit(X[bin_4], Y)
print('bin_4 oob score :%.5f' %(bin_4_Model.oob_score_),'   LB_Public : 0.7790')
# submission if you want
'''# submits
X_Submit = df_test.drop(labels=['PassengerId'],axis=1)

bin_4_pred = bin_4_Model.predict(X_Submit[bin_4])

submit = pd.DataFrame({"PassengerId": df_test['PassengerId'],
                      "Survived":bin_4_pred.astype(int)})
submit.to_csv("../output/submit_bin_4.csv",index=False)''';
compare = ['Sex_Code','Pclass','FareBin_Code_4','FareBin_Code_5','FareBin_Code_6']
selector = RFECV(RandomForestClassifier(n_estimators=250,min_samples_split=20),cv=10,n_jobs=-1)
selector.fit(X[compare], Y)
print(selector.support_)
print(selector.ranking_)
print(selector.grid_scores_*100)
score_b4,score_b5, score_b6 = [], [], []
seeds = 7 # kaggle kernel is unstable if you use too much computational resource
# I set seed = 10 normally when coding with my jupyter-notebook
for i in range(seeds):
    diff_cv = StratifiedKFold(n_splits=10,shuffle=True,random_state=i)
    selector = RFECV(RandomForestClassifier(random_state=i,n_estimators=250,min_samples_split=20),cv=diff_cv,n_jobs=-1)
    selector.fit(X[compare], Y)
    score_b4.append(selector.grid_scores_[2])
    score_b5.append(selector.grid_scores_[3])
    score_b6.append(selector.grid_scores_[4])
# to np.array
score_list = [score_b4, score_b5, score_b6]
for item in score_list:
    item = np.array(item*100)
# plot
fig = plt.figure(figsize= (18,8) )
ax = plt.gca()
ax.plot(range(seeds), score_b4,'-ok',label='bins = 4')
ax.plot(range(seeds), score_b5,'-og',label='bins = 5')
ax.plot(range(seeds), score_b6,'-ob',label='bins = 6')
ax.set_xlabel("Seed #", fontsize = '14')
ax.set_ylim(0.783,0.815)
ax.set_ylabel("Accuracy", fontsize = '14')
ax.set_title('bins = 4 vs bins = 5 vs bins = 6', fontsize='20')
plt.legend(fontsize = 14,loc='upper right')
b4, b5, b6 = ['Sex_Code', 'Pclass','FareBin_Code_4'], ['Sex_Code','Pclass','FareBin_Code_5'], ['Sex_Code','Pclass','FareBin_Code_6']
b4_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
b4_Model.fit(X[b4], Y)
b5_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
b5_Model.fit(X[b5], Y)
b6_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
b6_Model.fit(X[b6], Y)
print('b4 oob score :%.5f' %(b4_Model.oob_score_),'   LB_Public : 0.7790')
print('b5 oob score :%.5f '%(b5_Model.oob_score_),' LB_Public : 0.79425')
print('b6 oob score : %.5f' %(b6_Model.oob_score_), '  LB_Public : 0.77033')
# submission if you want
'''# submits
X_Submit = df_test.drop(labels=['PassengerId'],axis=1)

bin_5_pred = bin_5_Model.predict(X_Submit[bin_5])

submit = pd.DataFrame({"PassengerId": df_test['PassengerId'],
                      "Survived":bin_5_pred.astype(int)})
submit.to_csv("submit_bin_5.csv",index=False)''';
# In-Sample correct
b5_correct = Correct_classified_df(b5_Model, X[b5], Y)
# Compare 
tem = [df_data, Base_correct, b5_correct]
tem_factor_df = []
for i in tem:
    tem_factor_df.append(pd.pivot_table( i, values='Survived',index='Sex_Code',columns='Pclass').round(3))
# display
display(pd.concat([ tem_factor_df[0], tem_factor_df[1], tem_factor_df[2] ],keys=['Data', 'Base','Adding_FareBin'], axis = 1))
df_train['Ticket'].describe()
# count - unique != 0 which means there are duplicated
# Family_size
df_data['Family_size'] = df_data['SibSp'] + df_data['Parch'] + 1
# how about the fare values of deplicate tickets 
deplicate_ticket = []
for tk in df_data.Ticket.unique():
    tem = df_data.loc[df_data.Ticket == tk, 'Fare']
    #print(tem.count())
    if tem.count() > 1:
        #print(df_data.loc[df_data.Ticket == tk,['Name','Ticket','Fare']])
        deplicate_ticket.append(df_data.loc[df_data.Ticket == tk,['Name','Ticket','Fare','Cabin','Family_size','Survived']])
deplicate_ticket = pd.concat(deplicate_ticket)
deplicate_ticket.head(8)
print('people keep the same ticket: %.0f '%len(deplicate_ticket))
print('friends: %.0f '%len(deplicate_ticket[deplicate_ticket.Family_size == 1]))
print('families: %.0f '%len(deplicate_ticket[deplicate_ticket.Family_size > 1]))
df_fri = deplicate_ticket.loc[(deplicate_ticket.Family_size == 1) & (deplicate_ticket.Survived.notnull())].head(7)
# add a title friends
df_fami = deplicate_ticket.loc[(deplicate_ticket.Family_size > 1) & (deplicate_ticket.Survived.notnull())].head(7)
# add a title families
display(df_fri,df_fami)
# the same ticket family or friends
df_data['Connected_Survival'] = 0.5 # default 
for _, df_grp in df_data.groupby('Ticket'):
    if (len(df_grp) > 1):
        for ind, row in df_grp.iterrows():
            smax = df_grp.drop(ind)['Survived'].max()
            smin = df_grp.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                df_data.loc[df_data['PassengerId'] == passID, 'Connected_Survival'] = 1
            elif (smin==0.0):
                df_data.loc[df_data['PassengerId'] == passID, 'Connected_Survival'] = 0
# prints
print('people keep the same ticket: %.0f '%len(deplicate_ticket))
print('friends: %.0f '%len(deplicate_ticket[deplicate_ticket.Family_size == 1]))
print('families: %.0f '%len(deplicate_ticket[deplicate_ticket.Family_size > 1]))
print("people have connected information : %.0f" %(df_data[df_data['Connected_Survival']!=0.5].shape[0]))
df_data.groupby('Connected_Survival')['Survived'].mean().round(3)
# splits again beacuse we just engineered new feature
df_train = df_data[:len(df_train)]
df_test = df_data[len(df_train):]
# Training set and labels
X = df_train.drop(labels=['Survived','PassengerId'],axis=1)
Y = df_train['Survived']
connect = ['Sex_Code','Pclass','FareBin_Code_5','Connected_Survival']
connect_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
connect_Model.fit(X[connect], Y)
print('connect oob score :%.5f' %(connect_Model.oob_score_),'   LB_Public : 0.80832')
# submission if you want
'''# submits
X_Submit = df_test.drop(labels=['PassengerId'],axis=1)

connect_pred = connect_Model.predict(X_Submit[connect])

submit = pd.DataFrame({"PassengerId": df_test['PassengerId'],
                      "Survived":connect_pred.astype(int)})
submit.to_csv("submit_connect.csv",index=False)''';
# In-Sample correct
connect_correct = Correct_classified_df(connect_Model, X[connect], Y)
# Compare 
tem = [df_data, Base_correct, b5_correct,connect_correct]
tem_factor_df = []
for i in tem:
    tem_factor_df.append(pd.pivot_table( i, values='Survived',index='Sex_Code',columns='Pclass').round(3))
# display
keys=['Data', 'Base','Adding_FareBin','Adding_Connected']
display(pd.concat([ tem_factor_df[0], tem_factor_df[1], tem_factor_df[2],tem_factor_df[3] ],keys=keys, axis = 1))
# extracted title using name
df_data['Title'] = df_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df_data['Title'] = df_data['Title'].replace(['Capt', 'Col', 'Countess', 'Don',
                                               'Dr', 'Dona', 'Jonkheer', 
                                                'Major','Rev','Sir'],'Rare') 
df_data['Title'] = df_data['Title'].replace(['Mlle', 'Ms','Mme'],'Miss')
df_data['Title'] = df_data['Title'].replace(['Lady'],'Mrs')
df_data['Title'] = df_data['Title'].map({"Mr":0, "Rare" : 1, "Master" : 2,"Miss" : 3, "Mrs" : 4 })
Ti_pred = df_data.groupby('Title')['Age'].median().values
df_data['Ti_Age'] = df_data['Age']
# Filling the missing age
for i in range(0,5):
 # 0 1 2 3 4 5
    df_data.loc[(df_data.Age.isnull()) & (df_data.Title == i),'Ti_Age'] = Ti_pred[i]
df_data['Ti_Age'] = df_data['Ti_Age'].astype('int')

# extract minor
df_data['Age_copy'] = df_data['Age'].fillna(-1)
df_data['Minor'] = (df_data['Age_copy'] < 14.0) & (df_data['Age_copy']>= 0)
df_data['Minor'] = df_data['Minor'] * 1
# We could capture more 8 Master in Pclass = 3 by filling missing age 
df_data['Ti_Minor'] = ((df_data['Ti_Age']) < 14.0) * 1
print('The # of masters we found in missing Age by Title : ', (df_data['Ti_Minor'] - df_data['Minor']).sum())
# splits again beacuse we just engineered new feature
df_train = df_data[:len(df_train)]
df_test = df_data[len(df_train):]
# Training set and labels
X = df_train.drop(labels=['Survived','PassengerId'],axis=1)
Y = df_train['Survived']
# Show columns
X.columns
minor = ['Sex_Code','Pclass','FareBin_Code_5','Connected_Survival','Ti_Minor']
minor_Model = RandomForestClassifier(random_state=2,n_estimators=250,min_samples_split=20,oob_score=True)
minor_Model.fit(X[minor], Y)
print('Minor oob score :%.5f' %(minor_Model.oob_score_),'   LB_Public : 0.82296')
X_Submit = df_test.drop(labels=['PassengerId'],axis=1)

minor_pred = minor_Model.predict(X_Submit[minor])

submit = pd.DataFrame({"PassengerId": df_test['PassengerId'],
                      "Survived":minor_pred.astype(int)})
submit.to_csv("submit_minor.csv",index=False)
# In-Sample correct
minor_correct = Correct_classified_df(minor_Model, X[minor], Y)
# Compare 
tem = [df_data, Base_correct, b5_correct,connect_correct,minor_correct]
tem_factor_df = []
for i in tem:
    tem_factor_df.append(pd.pivot_table( i, values='Survived',index='Sex_Code',columns='Pclass').round(3))
# display
keys=['Data', 'Base','Adding_FareBin','Adding_Connected','Adding_Ti_Minor']
display(pd.concat([ tem_factor_df[0], tem_factor_df[1], tem_factor_df[2],tem_factor_df[3],tem_factor_df[4] ],keys=keys, axis = 1))