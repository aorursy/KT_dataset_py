import numpy as np

import seaborn as sns #数据可视化



import pandas as pd

from pandas.plotting import scatter_matrix



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

%matplotlib inline



import scipy as sp



import sklearn

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

import xgboost

from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics #评估标准



import IPython

from IPython import display



import random

import time

import warnings

warnings.filterwarnings('ignore')



#设置默认值

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
train_data=pd.read_csv('../input/train.csv')

test_data=pd.read_csv('../input/test.csv')

print(train_data.info()) #查看数据信息，包括数据取值、是否有空值、变量类型、数据大小等

print(' ')

print(test_data.info())
#数据缺失情况



print('the number of missing values in train_data: \n',train_data.isnull().sum())

print(' ')

print('the number of missing values in test_data: \n',test_data.isnull().sum())
#缺失值处理



data_cleaner=[train_data,test_data]

for dataset in data_cleaner:

    dataset['Age'].fillna(dataset['Age'].median(),inplace=True) #数值型，尝试使用中位数原地填充NA值

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True) #数值型变量

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True) #object类型，使用众数（众数有多个时，选择第一个）填充



# #删除无关变量列

# drop_columns=['PassengerId','Ticket','Cabin']

# train_data.drop(drop_columns, axis=1, inplace=True)

# test_data.drop(drop_columns, axis=1, inplace=True)
#特征工程，创造新的特征



for dataset in data_cleaner:

    dataset['family_size']=dataset['SibSp']+dataset['Parch']+1

    dataset['isalone']=1 #初始化为1，是alone

    dataset['isalone'][dataset[dataset['family_size']>1].index.tolist()]=0 #选取特定列/行

    dataset['Title'] = dataset['Name'].str.split(",",expand=True)[1].str.split(".",expand=True)[0]

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4) #等频分箱，每个小区间包含相等数量的数据；FareBin列的取值为不同的区间，区间长度不一定相同

    dataset['AgeBin']=pd.cut(dataset['Age'].astype(int), 5) #等距分箱，每个小区间的长度相等



print(train_data['Title'].value_counts())

print(' ')

minval=10

title_names=(train_data['Title'].value_counts()<minval)

train_data['Title']=train_data['Title'].apply(lambda x:'Misc' if title_names.loc[x]==True else x)

print(train_data['Title'].value_counts())
train_data.info()
#变量转换



#将object对象转为分类变量，并将分类变量转为标签（索引）表示

label=LabelEncoder()

for dataset in data_cleaner:

    dataset['sex_code']=label.fit_transform(dataset['Sex'])

    dataset['embarked_code']=label.fit_transform(dataset['Embarked'])

    dataset['title_code']=label.fit_transform(dataset['Title'])

    dataset['farebin_code']=label.fit_transform(dataset['FareBin'])

    dataset['agebin_code']=label.fit_transform(dataset['AgeBin'])

    

    

target=['Survived']

train_x_bin=['Pclass','Sex','AgeBin','family_size','FareBin','Embarked','Title','isalone']

print('train_xy_bin:{}'.format(train_x_bin+target))

print(' ')

train_x_code=['Pclass','sex_code','agebin_code','family_size','farebin_code','embarked_code','title_code','isalone']

train_x_dummy=pd.get_dummies(train_data[train_x_bin])

print('train_xy_dummy:{}'.format(train_x_dummy.columns.tolist()+target))

#绘制数值型变量的箱线图，以及在目标值上的堆叠直方图



plt.figure(figsize=[16,12])



plt.subplot(231)

plt.boxplot(x=train_data['Fare'],showmeans=True,meanline=True)

plt.title('Fare BoxPlot')

plt.ylabel('Fare ($)')



plt.subplot(232)

plt.boxplot(train_data['Age'],showmeans=True,meanline=True)

plt.title('Age BoxPlot')

plt.ylabel('Age')



plt.subplot(233)

plt.boxplot(train_data['family_size'],showmeans=True,meanline=True)

plt.title('family_size BoxPlot')

plt.ylabel('family_size')



plt.subplot(234)

plt.hist(x=[train_data[train_data['Survived']==1]['Fare'],train_data[train_data['Survived']==0]['Fare']], stacked=True, color=['g','r'],

         label=['Survived','Dead'])

plt.title('Fare Histogram by Survived')

plt.xlabel('Fare')

plt.ylabel('state of passenger')

plt.legend()



plt.subplot(235)

plt.hist(x=[train_data[train_data['Survived']==1]['Age'],train_data[train_data['Survived']==0]['Age']],stacked=True,

        color=['g','r'],label=['Survival','Dead'])

plt.title('Age Histogram by Survived')

plt.xlabel('Age')

plt.ylabel('state of passenger')

plt.legend()



plt.subplot(236)

plt.hist(x=[train_data[train_data['Survived']==1]['family_size'],train_data[train_data['Survived']==0]['family_size']],stacked=True,

        color=['g','r'],label=['Survival','Dead'])

plt.title('family_size Histogram by Survived')

plt.xlabel('family_size')

plt.ylabel('state of passenger')

plt.legend()
#绘制不同变量关于目标值的条形图



fig,saxis=plt.subplots(3,3,figsize=(16,12))

sns.barplot(x=train_data['Pclass'],y=train_data['Survived'],ax=saxis[0,0])

sns.barplot(x=train_data['Sex'],y=train_data['Survived'],ax=saxis[0,1])

sns.barplot(x=train_data['Embarked'],y=train_data['Survived'],ax=saxis[0,2])

sns.barplot(x=train_data['Title'],y=train_data['Survived'],ax=saxis[1,0])

sns.barplot(x=train_data['isalone'],y=train_data['Survived'],ax=saxis[1,1])

sns.barplot(x=train_data['family_size'],y=train_data['Survived'],ax=saxis[1,2])

sns.barplot(x=train_data['AgeBin'],y=train_data['Survived'],ax=saxis[2,0])

sns.barplot(x=train_data['FareBin'],y=train_data['Survived'],ax=saxis[2,1])
#可以知道：阶层（Pclass）和性别（Sex）对结果（Survived）很重要
#比较阶层与第二个特征的关系

#绘制箱线图，比较不同阶层有关其他变量的分布，以survived标记颜色



fig,(axis1,axis2,axis3)=plt.subplots(1,3,figsize=(16,8))



sns.boxplot(x='Pclass',y='Fare',hue='Survived',data=train_data,ax=axis1)

axis1.set_title('Pclass VS Fare Survived Comparison')



sns.boxplot(x='Pclass',y='Age',hue='Survived',data=train_data,ax=axis2)

axis2.set_title('Pclass VS Age Survived Comparison')



sns.boxplot(x='Pclass',y='family_size',hue='Survived',data=train_data,ax=axis3)

axis3.set_title('Pclass VS family_size Survived Comparison')
#比较性别与第二个特征对结果的综合影响



fig,axises=plt.subplots(2,2,figsize=(12,8))



sns.barplot(x='Sex',y='Survived',hue='Pclass',data=train_data,ax=axises[0,0])

sns.barplot(x='Sex',y='Survived',hue='Embarked',data=train_data,ax=axises[0,1])

sns.barplot(x='Sex',y='Survived',hue='isalone',data=train_data,ax=axises[1,0])

sns.barplot(x='Sex',y='Survived',hue='Title',data=train_data,ax=axises[1,1])
#绘制多区域网格，用于表示多个变量间的关系



a=sns.FacetGrid(train_data,hue='Survived',aspect=3) #aspect表示纵横比

a.map(sns.kdeplot,'Age',shade=True) #绘制变量核密度估计值

a.set(xlim=(0,max(train_data['Age'])))

a.add_legend()
b=sns.FacetGrid(train_data,row='Sex',col='Pclass',hue='Survived')

b.map(plt.hist,'Age',alpha=0.5)

b.add_legend()
def correlation_heatmap(df):

    """

    功能：绘制相关性热力图

    """

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True) #在两个颜色间制作一个发散的调色板

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True,  #在每个单元格内写入数据

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(train_data)
train_x,test_x,train_y,test_y=model_selection.train_test_split(train_data[train_x_code],train_data[target],random_state=0)
MLA = [

    #集成算法

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    XGBClassifier(), #XGBoost



    

    #广义线性模型

    linear_model.LogisticRegressionCV(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(), #用随机梯度下降算法训练的线性分类器的集合，如SVM（hinge损失）、logistic（log损失）

    linear_model.Perceptron(), #感知机模型

    

    #朴素贝叶斯

    naive_bayes.BernoulliNB(), #先验分布为伯努利分布，适用二元离散

    naive_bayes.GaussianNB(), #先验分布为高斯分布，适用连续

    naive_bayes.MultinomialNB(), #先验分布为多项式，适用多元离散

    

    #最近邻

    neighbors.KNeighborsClassifier(), #knn

    

    #SVM

    svm.SVC(probability=True),

    svm.LinearSVC(),

    

    #决策树

    tree.DecisionTreeClassifier(),

    

    #判别分析

    discriminant_analysis.LinearDiscriminantAnalysis(), #线性判别分析（LDA）

    discriminant_analysis.QuadraticDiscriminantAnalysis() #二次判别分析

    ]





#随机排列交叉验证；进行10次迭代，测试集和验证集比例分别为30%和60%，留出10%

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )





#定义一个表格，用于评价不同的机器学习算法

MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*std' ,'MLA Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)





#定义一个表格，用于表示不同机器学习算法在训练集上的预测值

MLA_predict = train_data[target]



#填充表格MLA_compare和MLA_predict

for row_index,mla in enumerate(MLA):

    """

    目的：

        生成一个包含机器学习算法名、参数、训练集精度、测试集精度、测试集精度3倍标准差以及平均拟合时间的表格，用于比较不同算法的优劣

        生成不同算法在训练集上的预测值

    """

    MLA_name=mla.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(mla.get_params())

    

    #交叉验证评估模型，记录得分次数。cv表示交叉验证分割策略

    cv_results = model_selection.cross_validate(mla, train_data[train_x_code], train_data[target], cv  = cv_split, return_train_score=True)



    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()

    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   

    #如果是非偏倚随机样本，mean+-3*std应包含99.74%的样本

    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*std'] = cv_results['test_score'].std()*3 #最坏情况

    



    #保存不同机器学习算法的预测结果

    mla.fit(train_data[train_x_code], train_data[target])

    MLA_predict[MLA_name] = mla.predict(train_data[train_x_code])



    

#按测试集精度排序

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

MLA_compare
MLA_predict.head()
#绘制不同算法在测试集上的精确度条形图



sns.barplot(x='MLA Test Accuracy Mean',y='MLA Name',data=MLA_compare,color='m')

plt.title('Machine Learning Algorithm Accuracy(%) \n')

plt.xlabel('Test Accuracy(%)')

plt.ylabel('Algorithm')
"""base_model"""



dtree=tree.DecisionTreeClassifier(random_state=0)

base_results=model_selection.cross_validate(dtree,train_data[train_x_code],train_data[target],cv=cv_split,return_train_score=True)

dtree.fit(train_data[train_x_code],train_data[target])

print('base_model feature: ',train_data[train_x_code].columns.values)

print('base_model parameters: ',dtree.get_params())

print('base_model train accuracy : {:.2f}'.format(base_results['train_score'].mean()*100))

print('base_model test accuracy : {:.2f}'.format(base_results['test_score'].mean()*100))

print(' ')
"""调参，模型优化"""



param_grid = {'criterion': ['gini', 'entropy'],

              'max_depth': [2,4,6,8,10,None],

              'random_state': [0]

             }

#对分类器的指定参数值进行网格搜索

adjusted_model=model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, 

                                            scoring='roc_auc', cv=cv_split, return_train_score=True)

adjusted_model.fit(train_data[train_x_code],train_data[target])



print('adjusted_model best parameters: ',adjusted_model.best_params_)

print('adjusted_model train accuracy: {:.2f}'.format(adjusted_model.cv_results_['mean_train_score'][adjusted_model.best_index_]*100))

print('adjusted model test accuracy: {:.2f}'.format(adjusted_model.cv_results_['mean_test_score'][adjusted_model.best_index_]*100))
"""特征选择，优化模型"""



#通过递归的消除特征和交叉验证，以获得具有最优数量的特征排序

#recursive feature elimination cross validate

rfe_model=feature_selection.RFECV(tree.DecisionTreeClassifier(), cv=cv_split, scoring='accuracy')

rfe_model.fit(train_data[train_x_code],train_data[target])

X_rfe=train_data[train_x_code].columns.values[rfe_model.support_]

rfe_results=model_selection.cross_validate(tree.DecisionTreeClassifier(), 

                                           train_data[X_rfe], train_data[target], cv=cv_split, return_train_score=True)



print('RFE model feature: ',X_rfe)

print('RFE model train accuracy: {:.2f}'.format(rfe_results['train_score'].mean()*100))

print('RFE model test accuracy: {:.2f}'.format(rfe_results['test_score'].mean()*100))
"""在特征选择的基础上进行调参，优化模型"""





rfe_adjusted_model=model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, 

                                            scoring='roc_auc', cv=cv_split, return_train_score=True)

rfe_adjusted_model.fit(train_data[X_rfe],train_data[target])



print('RFE adjusted_model best parameters: ',rfe_adjusted_model.best_params_)

print('RFE adjusted_model train accuracy: {:.2f}'.format(rfe_adjusted_model.cv_results_['mean_train_score'][rfe_adjusted_model.best_index_]*100))

print('RFE adjusted model test accuracy: {:.2f}'.format(rfe_adjusted_model.cv_results_['mean_test_score'][rfe_adjusted_model.best_index_]*100))
vote_estimator = [

    ('ada', ensemble.AdaBoostClassifier()),

    ('bc', ensemble.BaggingClassifier()),

    ('gbc', ensemble.GradientBoostingClassifier()),

    ('rfc', ensemble.RandomForestClassifier()),

    ('xgb', XGBClassifier()),

    ('lr', linear_model.LogisticRegressionCV()),   

    ('bnb', naive_bayes.BernoulliNB()),

    ('knn', neighbors.KNeighborsClassifier()),

    ('svc', svm.SVC(probability=True)),

]



#硬投票/多数表决

hard_vote_model=ensemble.VotingClassifier(estimators=vote_estimator, voting='hard')

hard_vote_results=model_selection.cross_validate(hard_vote_model, 

                                                 train_data[train_x_code], train_data[target], 

                                                 cv=cv_split, return_train_score=True)

hard_vote_model.fit(train_data[train_x_code], train_data[target])



print('hard_vote_model train accuracy: {:.2f}'.format(hard_vote_results['train_score'].mean()*100))

print('hard_vote_model test accuracy: {:.2f}'.format(hard_vote_results['test_score'].mean()*100))

print(' ')





#软投票/argmax

soft_vote_model=ensemble.VotingClassifier(estimators=vote_estimator, voting='soft')

soft_vote_results=model_selection.cross_validate(soft_vote_model, 

                                                 train_data[train_x_code], train_data[target], 

                                                 cv=cv_split, return_train_score=True)

soft_vote_model.fit(train_data[train_x_code], train_data[target])

print('soft_vote_model train accuracy: {:.2f}'.format(soft_vote_results['train_score'].mean()*100))

print('soft_vote_model test accuracy: {:.2f}'.format(soft_vote_results['test_score'].mean()*100))
grid_n_estimator = [10, 50, 100, 300] #基学习器数量

grid_ratio = [.1, .25, .5, .75, 1.0]

grid_learn = [.01, .03, .05, .1, .25] #学习率

grid_max_depth = [2, 4, 6, 8, 10, None] #

grid_min_samples = [5, 10, .03, .05, .10]

grid_criterion = ['gini', 'entropy']

grid_bool = [True, False]

grid_seed = [0]





grid_param = [

            [{

                #AdaBoostClassifier

                #base_estimator:None,DecisionTreeClassifier

                'n_estimators': grid_n_estimator,

                'learning_rate': grid_learn,

                'random_state': grid_seed

            }],

       

    

            [{

                #BaggingClassifier；基学习器为默认值，决策树

                'n_estimators': grid_n_estimator,

                'max_samples': grid_ratio, #从X中抽取的样本数量，训练基学习器的样本数量/比例

                'random_state': grid_seed

             }],





            [{

                #GradientBoostingClassifier

                'learning_rate': grid_learn,

                'n_estimators': grid_n_estimator,

                'max_depth': grid_max_depth, 

                'random_state': grid_seed

             }],



    

            [{

                #RandomForestClassifier

                'n_estimators': grid_n_estimator,

                'criterion': grid_criterion,

                'max_depth': grid_max_depth,

                'oob_score': [True], #out of bag

                'random_state': grid_seed

             }],

    

    

             [{

                #XGBClassifier

                'learning_rate': grid_learn,

                'max_depth': [1,2,4,6,8,10],

                'n_estimators': grid_n_estimator, 

                'seed': grid_seed

             }],

        

    

            [{

                #LogisticRegressionCV

                'fit_intercept': grid_bool,

                #'penalty': ['l1','l2'],

                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

                'random_state': grid_seed

             }],

            

    

            [{

                #BernoulliNB

                'alpha': grid_ratio

             }],



    

            [{

                #KNeighborsClassifier

                'n_neighbors': [1,2,3,4,5,6,7],

                'weights': ['uniform', 'distance'],

                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']

            }],

            

    

            [{

                #SVC

                'C': [1,2,3,4,5],

                #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], #默认是rbf（径向基函数）

                'gamma': grid_ratio,

                'decision_function_shape': ['ovo', 'ovr'],

                'probability': [True],

                'random_state': grid_seed

             }]

        ]







start_total = time.perf_counter() #性能计数器，seconds

for clf, param in zip (vote_estimator, grid_param):

    start = time.perf_counter()        

    best_search_model = model_selection.GridSearchCV(estimator = clf[1], 

                                                     param_grid = param, cv = cv_split, 

                                                     scoring = 'roc_auc', return_train_score=True)

    best_search_model.fit(train_data[train_x_code], train_data[target])

    run_time = time.perf_counter() - start



    best_param = best_search_model.best_params_

    print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run_time))

    print('{} train accuracy: {:.2f}'.format(clf[1].__class__.__name__, 

                                             best_search_model.cv_results_['mean_train_score'][best_search_model.best_index_]*100))

    print('{} test accuracy: {:.2f}'.format(clf[1].__class__.__name__, 

                                            best_search_model.cv_results_['mean_test_score'][best_search_model.best_index_]*100))

    print(' ')

    clf[1].set_params(**best_param) 

    



run_total = time.perf_counter() - start_total

print('Total optimization time was {:.2f} minutes.'.format(run_total/60))
gbdt=ensemble.GradientBoostingClassifier(learning_rate=0.05, max_depth=2, n_estimators=300, random_state=0)



# GBDT_model = model_selection.cross_validate(gbdt,train_data[train_x_code], train_data[target], cv = cv_split)



gbdt.fit(train_data[train_x_code], train_data[target])



test_data['Survived']=gbdt.predict(test_data[train_x_code])
submit=test_data[['PassengerId','Survived']]

submit.to_csv('submit.csv', index=False)

submit