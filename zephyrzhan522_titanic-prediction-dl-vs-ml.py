# Data Manipulation 
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
import missingno
import seaborn as sns

# Managing Warnings 
import warnings
warnings.filterwarnings('ignore')

# Plot the Figures Inline
%matplotlib inline

df_train=pd.read_csv('../input/titanic/train.csv')
df_test=pd.read_csv('../input/titanic/test.csv')
df_all = pd.concat([df_train, df_test]).reset_index(drop=True)
df_all.describe()
# 展示所有种类型特征
df_all.describe(include=['O'])
df_all.info()
df_all.head(5)
# 单特征展示Have a glance at data
import math
def plot_distribution(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataset.shape[1]) / cols)

    for i, column in enumerate(dataset.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        #ax.set_title(column)
        plt.xlabel(column, fontsize=20)
        plt.ylabel('',fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        if dataset.dtypes[column] == np.object:
            g = sns.countplot(y=column, data=dataset,palette='plasma')
            plt.xticks(rotation=25)
        else:
            #直方图，频数
            
            g = sns.distplot(dataset[column],kde_kws={'bw': 0.1})
            plt.ylabel(ylabel='Density',fontsize=20)
            plt.xticks(rotation=25)
            
    
plot_distribution(df_all[['Age','Cabin','Embarked','Fare','Parch','Pclass','Sex','SibSp','Survived']], cols=3, width=20, height=20, hspace=0.45, wspace=0.5)
fig = plt.figure(figsize=(15,8))
plt.subplot(2,2,1)
sns.violinplot(data=df_all, x="Sex", y="Age", hue="Survived",
               split=True, inner="quart", linewidth=1,
               palette={1: "#04A699", 0: ".85"})
sns.despine(left=True)
plt.subplot(2,2,2)
sns.violinplot(data=df_all, x="Sex", y="Pclass", hue="Survived",
               split=True, inner="quart", linewidth=1,
               palette={1: "#04A699", 0: ".85"})
sns.despine(left=True)
plt.subplot(2,2,3)
sns.violinplot(data=df_all, x="Sex", y="SibSp", hue="Survived",
               split=True, inner="quart", linewidth=1,
               palette={1: "#04A699", 0: ".85"})
sns.despine(left=True)
plt.subplot(2,2,4)
sns.violinplot(data=df_all, x="Sex", y="Parch", hue="Survived",
               split=True, inner="quart", linewidth=1,
               palette={1: "#04A699", 0: ".85"})
sns.despine(left=True)
s_pclass= df_all['Survived'].groupby(df_all['Pclass'])
s_pclass = s_pclass.value_counts().unstack()
s_pclass= s_pclass[[1.0,0.0]]
s_pclass.plot(kind='bar',stacked = True, colormap='tab20c')
s_sex = df_all['Survived'].groupby(df_all['Sex'])
s_sex = s_sex.value_counts().unstack()
s_sex = s_sex[[1.0,0.0]]
ax = s_sex.plot(kind='bar',stacked=True,colormap='tab20c')
sns.catplot(x="Pclass",y='Survived', hue="Sex", kind="point",
            palette="pastel", edgecolor=".6",
            data=df_all)

missingno.matrix(df_all, figsize = (20,5))
missingno.bar(df_all, sort='ascending', figsize = (20,5))
# 通过谷歌搜索唯二的两个乘客名字，可以得知登陆港口信息，不过这里我是引用gunesevitan的结论
df_all['Embarked'] = df_all['Embarked'].fillna('S')
df_all['Embarked'].head()
#对Cabin缺失值进行处理，利用U（Unknown）填充缺失值
df_all['Cabin']=df_all['Cabin'].fillna('U')
df_all['Cabin'].head()
#查看缺失值
df_all[df_all['Fare'].isnull()]
#假设船票价和Cabin,Pclass以及Embarked有关(按照常理推断)
df_all['Fare']=df_all['Fare'].fillna(df_all[(df_all['Pclass']==3)&(df_all['Embarked']=='S')&(df_all['Cabin']=='U')]['Fare'].mean())
#将Age完整的项作为训练集、将Age缺失的项作为测试集。
missing_age_df = df_all.iloc[:,[1,2,4,5,6,7,8,9,10,11]]
missing_age_df['Sex']= missing_age_df['Sex'].factorize()[0]
missing_age_df['Embarked']= missing_age_df['Embarked'].factorize()[0]
missing_age_df['Cabin']= missing_age_df['Cabin'].factorize()[0]
missing_age_df.corr()['Age'].sort_values(0)
missing_age_df = pd.DataFrame(missing_age_df[['Age', 'Parch','SibSp','Fare', 'Pclass','Cabin']])
#拆分训练集和测试集
age_train=missing_age_df[missing_age_df['Age'].notnull()]
age_test=missing_age_df[missing_age_df['Age'].isnull()]

#生成训练数据的特征和标签
age_train_X=age_train.drop(['Age'],axis=1)
age_train_y=age_train['Age']
#生成测试数据的特征
age_test_X=age_test.drop(['Age'],axis=1)

#利用随机森林构建模型
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
rfr.fit(age_train_X,age_train_y)

#模型得分
print('模型得分：',rfr.score(age_train_X,age_train_y))
#预测年龄
age_test_y=rfr.predict(age_test_X)
#填充预测数据
df_all.loc[df_all['Age'].isnull(),['Age']]=age_test_y
# 缺失值显示
missingno.matrix(df_all, figsize = (30,5))
df_all['Age'] = df_all['Age'].astype(int)
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(30,5)) 
sns.barplot(x="Age", y='Survived',data=df_all,palette='plasma');
df_all['Cabin'] = df_all['Cabin'].apply(lambda x:x[0])
sns.catplot(x="Cabin", kind="count",
            palette="pastel", edgecolor=".6",
            data=df_all)
df_all.loc[ (df_all.Cabin !='U'), 'Cabin' ] = "Yes"
df_all.loc[ (df_all.Cabin =='U'), 'Cabin' ] = "No"
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(15,6)) 
plt.subplot(1,2,1)
sns.countplot(x="Cabin",
            palette="pastel", edgecolor=".6",
            data=df_all)
plt.subplot(1,2,2)
sns.countplot(x="Cabin", hue="Survived",palette="pastel", edgecolor=".6",
            data=df_all)
df_base = df_all.iloc[:,1:]
df_base['Sex'] = df_base['Sex'].factorize()[0]
df_base = pd.get_dummies(df_base,columns={'Embarked','Pclass','Cabin'})
df_base.drop(['Name','Ticket'],axis=1,inplace=True)
from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler()

## transforming "age"
df_base['Age'] = std_scale.fit_transform(df_base['Age'].values.reshape(-1,1))
## transforming "fare"
df_base['Fare'] = std_scale.fit_transform(df_base['Fare'].values.reshape(-1,1))
df_base.head()
#分割训练集和测试集
df_base_train = df_base[df_base['Survived'].notnull()]
df_base_test = df_base[df_base['Survived'].isnull()]
# separating our independent and dependent variable
X_train = df_base_train.drop(['Survived'], axis = 1).astype(float)
y_train = df_base_train["Survived"].astype(float)

X_test = df_base_test.drop(['Survived'], axis = 1).astype(float)

#train model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='liblinear',C=1.0, penalty='l1', tol=1e-6)
clf.fit(X_train, y_train)
base_pred = clf.predict(X_test)
result = pd.DataFrame({'PassengerId':df_test['PassengerId'].values, 'Survived':base_pred.astype(np.int32)})
result.to_csv("baseline.csv", index=False)
result.head(10)
from sklearn.model_selection import learning_curve

train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = clf, X = X_train,
                                                   y = y_train, train_sizes = np.linspace(.05, 1., 10))
train_scores_mean = train_scores.mean(axis = 1)
validation_scores_mean = validation_scores.mean(axis = 1)
train_scores_std = train_scores.std(axis=1)
validation_scores_std = validation_scores.std(axis=1)

midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (validation_scores_mean[-1] - validation_scores_std[-1])) / 2
diff = (train_scores_mean[-1] + train_scores_std[-1]) - (validation_scores_mean[-1] - validation_scores_std[-1])
print("曲线中点：",midpoint,"Gap：",diff)

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,alpha=0.1)
plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.1)

plt.plot(train_sizes, train_scores_mean,'o-', label = 'Training score')
plt.plot(train_sizes, validation_scores_mean,'o-', label = 'Validation score')

plt.ylabel('Score', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves', fontsize = 18, y = 1.03)
plt.legend()
pd.DataFrame({"columns":list(df_base.columns)[1:], "coef":list(clf.coef_.T)})
df_improve = df_all.iloc[:,1:]
df_improve.head()
df_improve['Familysize'] = df_improve['Parch'] + df_improve['SibSp'] 
fg = plt.figure(figsize=(10,5))
sns.countplot(x='Familysize',hue='Survived',data=df_improve,palette="coolwarm")
aloneDiction = {}
aloneDiction[0] = 1
df_improve['isAlone'] = df_improve['Familysize'].map(aloneDiction).fillna(0)
fg = plt.figure(figsize=(8,5))
sns.countplot(x='isAlone',hue='Survived',data=df_improve,palette="coolwarm")
df_improve['Title'] = df_improve['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df_improve['Title'].value_counts()
TitleDict={}
TitleDict['Mr']='Man'
TitleDict['Mlle']='Woman'
TitleDict['Miss']='Woman'
TitleDict['Mme']='Woman'
TitleDict['Ms']='Woman'
TitleDict['Mrs']='Woman'
TitleDict['Master']='Boy'
TitleDict['Jonkheer']='Man'
TitleDict['Don']='Man'
TitleDict['Sir']='Man'
TitleDict['the Countess']='Woman'
TitleDict['Dona']='Woman'
TitleDict['Lady']='Woman'
TitleDict['Capt']='Man'
TitleDict['Col']='Man'
TitleDict['Major']='Man'
TitleDict['Dr']='Man'
TitleDict['Rev']='Man'

df_improve['Title']=df_improve['Title'].map(TitleDict)
df_improve['Title'].value_counts()
fg = plt.figure(figsize=(10,5))
sns.countplot(x='Title', hue='Survived',data=df_improve,palette="plasma")
plt.figure(figsize=(40,10))
plt.xticks(rotation=90)
plt.subplot(2,1,1)
sns.countplot(data=df_improve,x='Age',palette='plasma')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Age', fontsize=20)
plt.subplot(2,1,2)
sns.barplot(data=df_improve,x='Age', y='Survived',palette='plasma')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Age', fontsize=20)
#提取child年龄段
childGroup = df_improve.loc[df_improve['Age'] <= 12]
childGroup.value_counts('Survived')
sns.countplot(data=childGroup,x='Sex', hue='Survived',palette='plasma')
child_female = childGroup[(childGroup['Sex']=='male')& (childGroup['Survived']==1)].index
child_male = childGroup[(childGroup['Sex']=='female')& (childGroup['Survived']==0)].index
df_improve['Sex'][child_female]='female'
df_improve['Sex'][child_male]='male'
df_improve.drop(['Name','Ticket','SibSp','Parch','Age','Fare'],axis=1,inplace=True)
df_improve['Sex'] = df_improve['Sex'].factorize()[0]
df_improve['Cabin'] = df_improve['Cabin'].factorize()[0]

df_improve = pd.get_dummies(df_improve,columns={'Title','Pclass','Embarked','Familysize'})
plt.figure(figsize=(20,15))
sns.heatmap(df_improve.corr(),annot=True)
imp_train = df_improve[df_improve.Survived.notnull()]
final_test = df_improve[df_improve.Survived.isnull()].iloc[:,1:]
X_train = imp_train.iloc[:,1:]
y_train = imp_train.iloc[:,0]
seed = 2020
np.random.seed(seed)
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = round((X_train.shape[1]+1)/2), kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))

# Adding the second hidden layer
classifier.add(Dense(units = round((X_train.shape[1]+1)/2), kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
fit = classifier.fit(X_train, y_train, batch_size = 10, epochs = 300)
plt.plot(range(len(fit.history.get('loss'))),fit.history.get('loss'),label='Loss')
plt.plot(range(len(fit.history.get('accuracy'))),fit.history.get('accuracy'),label='Accuracy')
plt.legend()
# Predicting the Test set results
y_pred = classifier.predict(final_test)
y_pred = [0 if y<0.5 else 1 for y in y_pred]
y_pred = pd.DataFrame(y_pred)

result = pd.DataFrame({'PassengerId':df_test['PassengerId'].values, 'Survived':y_pred[0]})
result.to_csv("ANN-submission.csv", index=False)
result.head(10)
# Machine learning 
import sklearn.ensemble as ske
from sklearn import datasets, model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier

# Grid and Random Search
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

# Metrics
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
#设置kfold，交叉采样法拆分数据集
kfold=StratifiedKFold(n_splits=10)

#汇总不同模型算法
classifiers=[]
classifiers.append(SVC())
classifiers.append(DecisionTreeClassifier())
classifiers.append(RandomForestClassifier())
classifiers.append(ExtraTreesClassifier())
classifiers.append(GradientBoostingClassifier())
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())
classifiers.append(LGBMClassifier())
#不同机器学习交叉验证结果汇总
cv_results=[]
for classifiers in classifiers:
    cv_results.append(cross_val_score(classifiers,X_train,y_train,
                                      scoring='accuracy',cv=kfold,n_jobs=-1))
#求出模型得分的均值和标准差
cv_means=[]
cv_std=[]
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
#汇总数据
cvDf=pd.DataFrame({'cv_mean':cv_means,
                     'cv_std':cv_std,
                     'algorithm':['SVC','DecisionTreeCla','RandomForestCla','ExtraTreesCla',
                                  'GradientBoostingCla','KNN','LR','LGBMClassifier']})

cvDf = cvDf.sort_values('cv_mean',ascending=False)
cvDf
sns.barplot(data=cvDf,x='cv_mean',y='algorithm',**{'xerr':cv_std},palette='plasma')
#GradientBoostingClassifier模型
GBC = GradientBoostingClassifier(random_state=seed)
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [50,100,200],
              'learning_rate': [0.5, 0.1, 0.05, 0.01],
              'max_depth': [4,8,16],
              'min_samples_leaf': [100,150,200],
              'max_features': [0.3, 0.1] 
              }
modelgsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold,
                                     scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsGBC.fit(X_train,y_train)
modelgsGBC_accuracy = modelgsGBC.best_score_
modelgsGBC_parameters = modelgsGBC.best_params_

print(modelgsGBC_accuracy,modelgsGBC_parameters)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(modelgsGBC, X_train, y_train,
                                 cmap=plt.cm.Blues)
#train model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0,
                   class_weight=None,
                   dual=False, 
                   fit_intercept=True,
                   intercept_scaling=1, 
                   max_iter=100, 
                   multi_class='ovr',
                   n_jobs=-1,
                   penalty='l2',
                   random_state=seed, 
                   solver='liblinear',
                   tol=0.0001,
                   verbose=1, 
                   warm_start=False)
lr.fit(X_train, y_train)
disp2 = plot_confusion_matrix(lr, X_train, y_train,
                                 cmap=plt.cm.Blues)
from sklearn.metrics import roc_curve, auc
#plt.style.use('seaborn-pastel')
y_score1 = modelgsGBC.predict_proba(X_train)[:,1]
y_score2 = lr.predict_proba(X_train)[:,1]
FPR1, TPR1, _ = roc_curve(y_train, y_score1)
FPR2, TPR2, _ = roc_curve(y_train, y_score2)

ROC_AUC1 = auc(FPR1, TPR1)
ROC_AUC2 = auc(FPR2, TPR2)

plt.figure(figsize =[11,9])
plt.plot(FPR1, TPR1, label= 'GBC ROC curve(area = %0.2f)'%ROC_AUC1, linewidth= 4)
plt.plot(FPR2, TPR2, label= 'LR ROC curve(area = %0.2f)'%ROC_AUC2, linewidth= 4)
plt.legend()
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC for Titanic survivors', fontsize= 18)
y_pred=lr.predict(final_test)
y_pred=y_pred.astype(int)
result = pd.DataFrame({'PassengerId':df_test['PassengerId'].values, 'Survived':y_pred})
result.to_csv("GBC-submission.csv", index=False)
print(result['Survived'].value_counts())
result.head(10)