# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#load packages
import sys
print("Python version: {}". format(sys.version))

import pandas as pd
print("pandas version: {}". format(pd.__version__))

import matplotlib
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np
print("NumPy version: {}". format(np.__version__))

import scipy as sp
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display
print("IPython version: {}". format(IPython.__version__)) 

import sklearn
print("scikit-learn version: {}". format(sklearn.__version__))

import random
import time

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
# 加载数据源
data_raw = pd.read_csv('../input/train.csv')
data_val = pd.read_csv('../input/test.csv')

# 玩数据前，先copy一个备份
data1 = data_raw.copy(deep = True)
# 训练集和测试集都需要数据清洗，此处操作是为了后面清洗方便
data_cleaner = [data1, data_val]

# 预览数据
print (data1.info())
data1.sample(10)
print('训练集缺失值统计:\n', data1.isnull().sum())
print("-"*10)

print('测试集缺失值统计:\n', data_val.isnull().sum())
print("-"*10)

data_raw.describe(include = 'all')
### 缺失值处理
for dataset in data_cleaner:    
    #用中位数填充缺失年龄
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    #中位数填充票价
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    
# 删除乘客id、客舱号码与票号，因为这些数据感觉对预测乘客最终是否存活没有很大帮助
drop_column = ['PassengerId','Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace = True)

print(data1.isnull().sum())
print("-"*10)
print(data_val.isnull().sum())
### 特征工程
for dataset in data_cleaner:    
    #Discrete variables 离散变量
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1 #初始化为1
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 #如果家庭人数大于1，更新为0

    # http://www.pythonforbeginners.com/dictionary/python-split
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    # qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

#cleanup rare title names
#print(data1['Title'].value_counts())
stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
title_names = (data1['Title'].value_counts() < stat_min) #this will create a true false series with title name as index

#apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(data1['Title'].value_counts())
print("-"*10)

#再次预览数据
data1.info()
print("-"*10)
data_val.info()
print("-"*10)
data1.sample(10)
# 使用 Label Encoder 对数据集对象进行类别转换

label = LabelEncoder()
for dataset in data_cleaner:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])


# 定义 y
Target = ['Survived']

# 定义x
# data1_x 表格展示
# data1_x_calc 模型算法的输入
data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']
data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare']
data1_xy =  Target + data1_x
print('Original X Y: ', data1_xy, '\n')

#define x variables for original w/bin features to remove continuous variables
data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')


#define x and y variables for dummy features original
data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('Dummy X Y: ', data1_xy_dummy, '\n')


data1_dummy.head()

# 数据已经清洗完毕，现在重新检查下数据
print('训练集缺失值统计: \n', data1.isnull().sum())
print("-"*10)
print (data1.info())
print("-"*10)

print('验证集缺失值统计: \n', data_val.isnull().sum())
print("-"*10)
print (data_val.info())
print("-"*10)

data_raw.describe(include = 'all')
# https://www.quora.com/What-is-seed-in-random-number-generation
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target] , random_state = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)


print("Data1 Shape: {}".format(data1.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))

train1_x_bin.head()
#Discrete Variable Correlation by Survival using
#描述变量的相关性通过 
#group by aka pivot table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
for x in data1_x:
    if data1[x].dtype != 'float64' :
        print('Survival Correlation by:', x)
        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')
        

#using crosstabs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html
print(pd.crosstab(data1['Title'],data1[Target[0]]))
#注意: 特意用不同的形式进行绘画仅仅是基于学习的目的

#可供选择的绘图方式: https://pandas.pydata.org/pandas-docs/stable/visualization.html
#各种图形介绍：https://blog.csdn.net/suzyu12345/article/details/69029106

#这里将使用 matplotlib.pyplot: https://matplotlib.org/api/pyplot_api.html

#将使用 figure去组织图形: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure
#subplot: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot
#and subplotS: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html?highlight=matplotlib%20pyplot%20subplots#matplotlib.pyplot.subplots

#graph distribution of quantitative data
plt.figure(figsize=[16,12])

# 箱型图
plt.subplot(231)
plt.boxplot(x=data1['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(data1['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')

plt.subplot(233)
plt.boxplot(data1['FamilySize'], showmeans = True, meanline = True)
plt.title('Family Size Boxplot')
plt.ylabel('Family Size (#)')

# 直方图
plt.subplot(234)
plt.hist(x = [data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(236)
plt.hist(x = [data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()

#使用 seaborn 进行多变量的比较

#graph individual features by survival  通过生存来绘制个体特征
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

# 柱状图
sns.barplot(x = 'Embarked', y = 'Survived', data=data1, ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data1, ax = saxis[0,1])
sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data1, ax = saxis[0,2])

sns.pointplot(x = 'FareBin', y = 'Survived',  data=data1, ax = saxis[1,0])
sns.pointplot(x = 'AgeBin', y = 'Survived',  data=data1, ax = saxis[1,1])
sns.pointplot(x = 'FamilySize', y = 'Survived', data=data1, ax = saxis[1,2])
#graph distribution of qualitative data: Pclass
#we know class mattered in survival, now let's compare class and a 2nd feature
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))

sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = data1, ax = axis1)
axis1.set_title('Pclass vs Fare Survival Comparison')

# 小提琴图
sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data1, split = True, ax = axis2)
axis2.set_title('Pclass vs Age Survival Comparison')

sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = data1, ax = axis3)
axis3.set_title('Pclass vs Family Size Survival Comparison')
#graph distribution of qualitative data: Sex
#we know sex mattered in survival, now let's compare sex and a 2nd feature
fig, qaxis = plt.subplots(1,3,figsize=(14,12))

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=data1, ax = qaxis[0])
axis1.set_title('Sex vs Embarked Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=data1, ax  = qaxis[1])
axis1.set_title('Sex vs Pclass Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=data1, ax  = qaxis[2])
axis1.set_title('Sex vs IsAlone Survival Comparison')
#more side-by-side comparisons
#更多的比较
fig, (maxis1, maxis2) = plt.subplots(1, 2,figsize=(14,12))

#how does family size factor with sex & survival compare
sns.pointplot(x="FamilySize", y="Survived", hue="Sex", data=data1,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis1)

#how does class factor with sex & survival compare
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data1,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis2)
#how does embark port factor with class, sex, and survival compare
#facetgrid: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html
e = sns.FacetGrid(data1, col = 'Embarked')
e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep')
e.add_legend()
#plot distributions of age of passengers who survived or did not survive
# 绘制年龄的分布
a = sns.FacetGrid( data1, hue = 'Survived', aspect=4 )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0 , data1['Age'].max()))
a.add_legend()
#histogram comparison of sex, class, and age by survival
h = sns.FacetGrid(data1, row = 'Sex', col = 'Pclass', hue = 'Survived')
h.map(plt.hist, 'Age', alpha = .75)
h.add_legend()
#pair plots of entire dataset
# 整个数据集的 pair plots
pp = sns.pairplot(data1, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
pp.set(xticklabels=[])
#数据集的相关热图
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(data1)
# 机器学习算法选择及初始化
MLA = [
    #集成算法
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #高斯
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #贝叶斯
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis 判别分析
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]


#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
# ShuffleSplit 是 train_test_split 的替换方案
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = data1[Target]

#index through MLA and save performance to table
# 通过MLA进行索引，并将表现保存到表格中
row_index = 0
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, data1[data1_x_bin], data1[Target], cv  = cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    

    #save MLA predictions - see section 6 for usage
    alg.fit(data1[data1_x_bin], data1[Target])
    MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin])
    
    row_index+=1

    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
# 算法比对
MLA_compare

# 预测结果
# MLA_predict
#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html
sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')

#使用 pyplot 进行美化: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
#这里以决策树为基准模型
dtree = tree.DecisionTreeClassifier(random_state = 0)
base_results = model_selection.cross_validate(dtree, data1[data1_x_bin], data1[Target], cv  = cv_split)
dtree.fit(data1[data1_x_bin], data1[Target])

print('BEFORE DT Parameters: ', dtree.get_params())
print("BEFORE DT Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 
print("BEFORE DT Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))
print("BEFORE DT Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))
#print("BEFORE DT Test w/bin set score min: {:.2f}". format(base_results['test_score'].min()*100))
print('-'*10)

#tune hyper-parameters: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
param_grid = {'criterion': ['gini', 'entropy'],  #scoring methodology; two supported formulas for calculating information gain - default is gini
              #'splitter': ['best', 'random'], #splitting methodology; two supported strategies - default is best
              'max_depth': [2,4,6,8,10,None], #max depth tree can grow; default is none
              #'min_samples_split': [2,5,10,.03,.05], #minimum subset size BEFORE new split (fraction is % of total); default is 2
              #'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1
              #'max_features': [None, 'auto'], #max features to consider when performing split; default none or all
              'random_state': [0] #seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
             }

#print(list(model_selection.ParameterGrid(param_grid)))

#choose best model with grid_search: #http://scikit-learn.org/stable/modules/grid_search.html#grid-search
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)
tune_model.fit(data1[data1_x_bin], data1[Target])

#print(tune_model.cv_results_.keys())
#print(tune_model.cv_results_['params'])
print('AFTER DT Parameters: ', tune_model.best_params_)
#print(tune_model.cv_results_['mean_train_score'])
print("AFTER DT Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 
#print(tune_model.cv_results_['mean_test_score'])
print("AFTER DT Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print("AFTER DT Test w/bin score 3*std: +/- {:.2f}". format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
print('-'*10)

#duplicates gridsearchcv
#tune_results = model_selection.cross_validate(tune_model, data1[data1_x_bin], data1[Target], cv  = cv_split)

#print('AFTER DT Parameters: ', tune_model.best_params_)
#print("AFTER DT Training w/bin set score mean: {:.2f}". format(tune_results['train_score'].mean()*100)) 
#print("AFTER DT Test w/bin set score mean: {:.2f}". format(tune_results['test_score'].mean()*100))
#print("AFTER DT Test w/bin set score min: {:.2f}". format(tune_results['test_score'].min()*100))
#print('-'*10)

Y_pred = tune_model.predict(data_val[data1_x_bin])
submission = pd.DataFrame({
        "PassengerId": data_val["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)
#base model
print('BEFORE DT RFE Training Shape Old: ', data1[data1_x_bin].shape) 
print('BEFORE DT RFE Training Columns Old: ', data1[data1_x_bin].columns.values)

print("BEFORE DT RFE Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 
print("BEFORE DT RFE Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))
print("BEFORE DT RFE Test w/bin score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))
print('-'*10)



#feature selection
dtree_rfe = feature_selection.RFECV(dtree, step = 1, scoring = 'accuracy', cv = cv_split)
dtree_rfe.fit(data1[data1_x_bin], data1[Target])

#transform x&y to reduced features and fit new model
#alternative: can use pipeline to reduce fit and transform steps: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
X_rfe = data1[data1_x_bin].columns.values[dtree_rfe.get_support()]
rfe_results = model_selection.cross_validate(dtree, data1[X_rfe], data1[Target], cv  = cv_split)

#print(dtree_rfe.grid_scores_)
print('AFTER DT RFE Training Shape New: ', data1[X_rfe].shape) 
print('AFTER DT RFE Training Columns New: ', X_rfe)

print("AFTER DT RFE Training w/bin score mean: {:.2f}". format(rfe_results['train_score'].mean()*100)) 
print("AFTER DT RFE Test w/bin score mean: {:.2f}". format(rfe_results['test_score'].mean()*100))
print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}". format(rfe_results['test_score'].std()*100*3))
print('-'*10)


#tune rfe model
rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)
rfe_tune_model.fit(data1[X_rfe], data1[Target])

#print(rfe_tune_model.cv_results_.keys())
#print(rfe_tune_model.cv_results_['params'])
print('AFTER DT RFE Tuned Parameters: ', rfe_tune_model.best_params_)
#print(rfe_tune_model.cv_results_['mean_train_score'])
print("AFTER DT RFE Tuned Training w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 
#print(rfe_tune_model.cv_results_['mean_test_score'])
print("AFTER DT RFE Tuned Test w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print("AFTER DT RFE Tuned Test w/bin score 3*std: +/- {:.2f}". format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
print('-'*10)
#Graph MLA version of Decision Tree: http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
#决策树可视化
import graphviz 
dot_data = tree.export_graphviz(dtree, out_file=None, 
                                feature_names = data1_x_bin, class_names = True,
                                filled = True, rounded = True)
graph = graphviz.Source(dot_data) 
# graph
#compare algorithm predictions with each other, where 1 = exactly similar and 0 = exactly opposite
#there are some 1's, but enough blues and light reds to create a "super algorithm" by combining them
# 将算法的预测结果两两比对
correlation_heatmap(MLA_predict)
#why choose one model, when you can pick them all with voting classifier
#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
#removed models w/o attribute 'predict_proba' required for vote classifier and models with a 1.0 correlation to another model
vote_est = [
    #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
    ('ada', ensemble.AdaBoostClassifier()),
    ('bc', ensemble.BaggingClassifier()),
    ('etc',ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),

    #Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc
    ('gpc', gaussian_process.GaussianProcessClassifier()),
    
    #GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ('lr', linear_model.LogisticRegressionCV()),
    
    #Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html
    ('bnb', naive_bayes.BernoulliNB()),
    ('gnb', naive_bayes.GaussianNB()),
    
    #Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html
    ('knn', neighbors.KNeighborsClassifier()),
    
    #SVM: http://scikit-learn.org/stable/modules/svm.html
    ('svc', svm.SVC(probability=True)),
    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
   ('xgb', XGBClassifier())

]


#Hard Vote or majority rules
vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
vote_hard_cv = model_selection.cross_validate(vote_hard, data1[data1_x_bin], data1[Target], cv  = cv_split)
vote_hard.fit(data1[data1_x_bin], data1[Target])

print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 
print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))
print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))
print('-'*10)


#Soft Vote or weighted probabilities
vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
vote_soft_cv = model_selection.cross_validate(vote_soft, data1[data1_x_bin], data1[Target], cv  = cv_split)
vote_soft.fit(data1[data1_x_bin], data1[Target])

print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 
print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))
print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))
print('-'*10)

# #IMPORTANT: THIS SECTION IS UNDER CONSTRUCTION!!!! 12.24.17
# #UPDATE: This section was scrapped for the next section; as it's more computational friendly.

# #WARNING: Running is very computational intensive and time expensive
# #code is written for experimental/developmental purposes and not production ready


# #tune each estimator before creating a super model
# #http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# grid_n_estimator = [50,100,300]
# grid_ratio = [.1,.25,.5,.75,1.0]
# grid_learn = [.01,.03,.05,.1,.25]
# grid_max_depth = [2,4,6,None]
# grid_min_samples = [5,10,.03,.05,.10]
# grid_criterion = ['gini', 'entropy']
# grid_bool = [True, False]
# grid_seed = [0]

# vote_param = [{
# #            #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
#             'ada__n_estimators': grid_n_estimator,
#             'ada__learning_rate': grid_ratio,
#             'ada__algorithm': ['SAMME', 'SAMME.R'],
#             'ada__random_state': grid_seed,
    
#             #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
#             'bc__n_estimators': grid_n_estimator,
#             'bc__max_samples': grid_ratio,
#             'bc__oob_score': grid_bool, 
#             'bc__random_state': grid_seed,
            
#             #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
#             'etc__n_estimators': grid_n_estimator,
#             'etc__criterion': grid_criterion,
#             'etc__max_depth': grid_max_depth,
#             'etc__random_state': grid_seed,


#             #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
#             'gbc__loss': ['deviance', 'exponential'],
#             'gbc__learning_rate': grid_ratio,
#             'gbc__n_estimators': grid_n_estimator,
#             'gbc__criterion': ['friedman_mse', 'mse', 'mae'],
#             'gbc__max_depth': grid_max_depth,
#             'gbc__min_samples_split': grid_min_samples,
#             'gbc__min_samples_leaf': grid_min_samples,      
#             'gbc__random_state': grid_seed,
    
#             #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
#             'rfc__n_estimators': grid_n_estimator,
#             'rfc__criterion': grid_criterion,
#             'rfc__max_depth': grid_max_depth,
#             'rfc__min_samples_split': grid_min_samples,
#             'rfc__min_samples_leaf': grid_min_samples,   
#             'rfc__bootstrap': grid_bool,
#             'rfc__oob_score': grid_bool, 
#             'rfc__random_state': grid_seed,
        
#             #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
#             'lr__fit_intercept': grid_bool,
#             'lr__penalty': ['l1','l2'],
#             'lr__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
#             'lr__random_state': grid_seed,
            
#             #http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
#             'bnb__alpha': grid_ratio,
#             'bnb__prior': grid_bool,
#             'bnb__random_state': grid_seed,
    
#             #http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
#             'knn__n_neighbors': [1,2,3,4,5,6,7],
#             'knn__weights': ['uniform', 'distance'],
#             'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#             'knn__random_state': grid_seed,
            
#             #http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
#             #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
#             'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#             'svc__C': grid_max_depth,
#             'svc__gamma': grid_ratio,
#             'svc__decision_function_shape': ['ovo', 'ovr'],
#             'svc__probability': [True],
#             'svc__random_state': grid_seed,
    
    
#             #http://xgboost.readthedocs.io/en/latest/parameter.html
#             'xgb__learning_rate': grid_ratio,
#             'xgb__max_depth': [2,4,6,8,10],
#             'xgb__tree_method': ['exact', 'approx', 'hist'],
#             'xgb__objective': ['reg:linear', 'reg:logistic', 'binary:logistic'],
#             'xgb__seed': grid_seed    

#         }]




# #Soft Vote with tuned models
# #grid_soft = model_selection.GridSearchCV(estimator = vote_soft, param_grid = vote_param, cv = 2, scoring = 'roc_auc')
# #grid_soft.fit(data1[data1_x_bin], data1[Target])

# #print(grid_soft.cv_results_.keys())
# #print(grid_soft.cv_results_['params'])
# #print('Soft Vote Tuned Parameters: ', grid_soft.best_params_)
# #print(grid_soft.cv_results_['mean_train_score'])
# #print("Soft Vote Tuned Training w/bin set score mean: {:.2f}". format(grid_soft.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 
# #print(grid_soft.cv_results_['mean_test_score'])
# #print("Soft Vote Tuned Test w/bin set score mean: {:.2f}". format(grid_soft.cv_results_['mean_test_score'][tune_model.best_index_]*100))
# #print("Soft Vote Tuned Test w/bin score 3*std: +/- {:.2f}". format(grid_soft.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
# #print('-'*10)


# #credit: https://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/
# #cv_keys = ('mean_test_score', 'std_test_score', 'params')
# #for r, _ in enumerate(grid_soft.cv_results_['mean_test_score']):
# #    print("%0.3f +/- %0.2f %r"
# #          % (grid_soft.cv_results_[cv_keys[0]][r],
# #             grid_soft.cv_results_[cv_keys[1]][r] / 2.0,
# #             grid_soft.cv_results_[cv_keys[2]][r]))


# #print('-'*10)

# #WARNING: Running is very computational intensive and time expensive.
# #Code is written for experimental/developmental purposes and not production ready!


# #Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# grid_n_estimator = [10, 50, 100, 300]
# grid_ratio = [.1, .25, .5, .75, 1.0]
# grid_learn = [.01, .03, .05, .1, .25]
# grid_max_depth = [2, 4, 6, 8, 10, None]
# grid_min_samples = [5, 10, .03, .05, .10]
# grid_criterion = ['gini', 'entropy']
# grid_bool = [True, False]
# grid_seed = [0]


# grid_param = [
#             [{
#             #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
#             'n_estimators': grid_n_estimator, #default=50
#             'learning_rate': grid_learn, #default=1
#             #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
#             'random_state': grid_seed
#             }],
       
    
#             [{
#             #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
#             'n_estimators': grid_n_estimator, #default=10
#             'max_samples': grid_ratio, #default=1.0
#             'random_state': grid_seed
#              }],

    
#             [{
#             #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
#             'n_estimators': grid_n_estimator, #default=10
#             'criterion': grid_criterion, #default=”gini”
#             'max_depth': grid_max_depth, #default=None
#             'random_state': grid_seed
#              }],


#             [{
#             #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
#             #'loss': ['deviance', 'exponential'], #default=’deviance’
#             'learning_rate': [.05], #default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
#             'n_estimators': [300], #default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
#             #'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
#             'max_depth': grid_max_depth, #default=3   
#             'random_state': grid_seed
#              }],

    
#             [{
#             #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
#             'n_estimators': grid_n_estimator, #default=10
#             'criterion': grid_criterion, #default=”gini”
#             'max_depth': grid_max_depth, #default=None
#             'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
#             'random_state': grid_seed
#              }],
    
#             [{    
#             #GaussianProcessClassifier
#             'max_iter_predict': grid_n_estimator, #default: 100
#             'random_state': grid_seed
#             }],
        
    
#             [{
#             #LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV
#             'fit_intercept': grid_bool, #default: True
#             #'penalty': ['l1','l2'],
#             'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs
#             'random_state': grid_seed
#              }],
            
    
#             [{
#             #BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
#             'alpha': grid_ratio, #default: 1.0
#              }],
    
    
#             #GaussianNB - 
#             [{}],
    
#             [{
#             #KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
#             'n_neighbors': [1,2,3,4,5,6,7], #default: 5
#             'weights': ['uniform', 'distance'], #default = ‘uniform’
#             'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
#             }],
            
    
#             [{
#             #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
#             #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
#             #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#             'C': [1,2,3,4,5], #default=1.0
#             'gamma': grid_ratio, #edfault: auto
#             'decision_function_shape': ['ovo', 'ovr'], #default:ovr
#             'probability': [True],
#             'random_state': grid_seed
#              }],

    
#             [{
#             #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html
#             'learning_rate': grid_learn, #default: .3
#             'max_depth': [1,2,4,6,8,10], #default 2
#             'n_estimators': grid_n_estimator, 
#             'seed': grid_seed  
#              }]   
#         ]



# start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter
# for clf, param in zip (vote_est, grid_param): #https://docs.python.org/3/library/functions.html#zip

#     #print(clf[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm
#     #print(param)
    
    
#     start = time.perf_counter()        
#     best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = 'roc_auc')
#     best_search.fit(data1[data1_x_bin], data1[Target])
#     run = time.perf_counter() - start

#     best_param = best_search.best_params_
#     print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))
#     clf[1].set_params(**best_param) 


# run_total = time.perf_counter() - start_total
# print('Total optimization time was {:.2f} minutes.'.format(run_total/60))

# print('-'*10)
# #Hard Vote or majority rules w/Tuned Hyperparameters
# grid_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
# grid_hard_cv = model_selection.cross_validate(grid_hard, data1[data1_x_bin], data1[Target], cv  = cv_split)
# grid_hard.fit(data1[data1_x_bin], data1[Target])

# print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100)) 
# print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))
# print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))
# print('-'*10)

# #Soft Vote or weighted probabilities w/Tuned Hyperparameters
# grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
# grid_soft_cv = model_selection.cross_validate(grid_soft, data1[data1_x_bin], data1[Target], cv  = cv_split)
# grid_soft.fit(data1[data1_x_bin], data1[Target])

# print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_soft_cv['train_score'].mean()*100)) 
# print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))
# print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_soft_cv['test_score'].std()*100*3))
# print('-'*10)


# #12/31/17 tuned with data1_x_bin
# #The best parameter for AdaBoostClassifier is {'learning_rate': 0.1, 'n_estimators': 300, 'random_state': 0} with a runtime of 33.39 seconds.
# #The best parameter for BaggingClassifier is {'max_samples': 0.25, 'n_estimators': 300, 'random_state': 0} with a runtime of 30.28 seconds.
# #The best parameter for ExtraTreesClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'random_state': 0} with a runtime of 64.76 seconds.
# #The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 34.35 seconds.
# #The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 76.32 seconds.
# #The best parameter for GaussianProcessClassifier is {'max_iter_predict': 10, 'random_state': 0} with a runtime of 6.01 seconds.
# #The best parameter for LogisticRegressionCV is {'fit_intercept': True, 'random_state': 0, 'solver': 'liblinear'} with a runtime of 8.04 seconds.
# #The best parameter for BernoulliNB is {'alpha': 0.1} with a runtime of 0.19 seconds.
# #The best parameter for GaussianNB is {} with a runtime of 0.04 seconds.
# #The best parameter for KNeighborsClassifier is {'algorithm': 'brute', 'n_neighbors': 7, 'weights': 'uniform'} with a runtime of 4.84 seconds.
# #The best parameter for SVC is {'C': 2, 'decision_function_shape': 'ovo', 'gamma': 0.1, 'probability': True, 'random_state': 0} with a runtime of 29.39 seconds.
# #The best parameter for XGBClassifier is {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 300, 'seed': 0} with a runtime of 46.23 seconds.
# #Total optimization time was 5.56 minutes.
# #prepare data for modeling
# print(data_val.info())
# print("-"*10)
# #data_val.sample(10)



# #handmade decision tree - submission score = 0.77990
# data_val['Survived'] = mytree(data_val).astype(int)


# #decision tree w/full dataset modeling submission score: defaults= 0.76555, tuned= 0.77990
# #submit_dt = tree.DecisionTreeClassifier()
# #submit_dt = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)
# #submit_dt.fit(data1[data1_x_bin], data1[Target])
# #print('Best Parameters: ', submit_dt.best_params_) #Best Parameters:  {'criterion': 'gini', 'max_depth': 4, 'random_state': 0}
# #data_val['Survived'] = submit_dt.predict(data_val[data1_x_bin])


# #bagging w/full dataset modeling submission score: defaults= 0.75119, tuned= 0.77990
# #submit_bc = ensemble.BaggingClassifier()
# #submit_bc = model_selection.GridSearchCV(ensemble.BaggingClassifier(), param_grid= {'n_estimators':grid_n_estimator, 'max_samples': grid_ratio, 'oob_score': grid_bool, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
# #submit_bc.fit(data1[data1_x_bin], data1[Target])
# #print('Best Parameters: ', submit_bc.best_params_) #Best Parameters:  {'max_samples': 0.25, 'n_estimators': 500, 'oob_score': True, 'random_state': 0}
# #data_val['Survived'] = submit_bc.predict(data_val[data1_x_bin])


# #extra tree w/full dataset modeling submission score: defaults= 0.76555, tuned= 0.77990
# #submit_etc = ensemble.ExtraTreesClassifier()
# #submit_etc = model_selection.GridSearchCV(ensemble.ExtraTreesClassifier(), param_grid={'n_estimators': grid_n_estimator, 'criterion': grid_criterion, 'max_depth': grid_max_depth, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
# #submit_etc.fit(data1[data1_x_bin], data1[Target])
# #print('Best Parameters: ', submit_etc.best_params_) #Best Parameters:  {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'random_state': 0}
# #data_val['Survived'] = submit_etc.predict(data_val[data1_x_bin])


# #random foreset w/full dataset modeling submission score: defaults= 0.71291, tuned= 0.73205
# #submit_rfc = ensemble.RandomForestClassifier()
# #submit_rfc = model_selection.GridSearchCV(ensemble.RandomForestClassifier(), param_grid={'n_estimators': grid_n_estimator, 'criterion': grid_criterion, 'max_depth': grid_max_depth, 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
# #submit_rfc.fit(data1[data1_x_bin], data1[Target])
# #print('Best Parameters: ', submit_rfc.best_params_) #Best Parameters:  {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'random_state': 0}
# #data_val['Survived'] = submit_rfc.predict(data_val[data1_x_bin])



# #ada boosting w/full dataset modeling submission score: defaults= 0.74162, tuned= 0.75119
# #submit_abc = ensemble.AdaBoostClassifier()
# #submit_abc = model_selection.GridSearchCV(ensemble.AdaBoostClassifier(), param_grid={'n_estimators': grid_n_estimator, 'learning_rate': grid_ratio, 'algorithm': ['SAMME', 'SAMME.R'], 'random_state': grid_seed}, scoring = 'roc_auc', cv = cv_split)
# #submit_abc.fit(data1[data1_x_bin], data1[Target])
# #print('Best Parameters: ', submit_abc.best_params_) #Best Parameters:  {'algorithm': 'SAMME.R', 'learning_rate': 0.1, 'n_estimators': 300, 'random_state': 0}
# #data_val['Survived'] = submit_abc.predict(data_val[data1_x_bin])


# #gradient boosting w/full dataset modeling submission score: defaults= 0.75119, tuned= 0.77033
# #submit_gbc = ensemble.GradientBoostingClassifier()
# #submit_gbc = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid={'learning_rate': grid_ratio, 'n_estimators': grid_n_estimator, 'max_depth': grid_max_depth, 'random_state':grid_seed}, scoring = 'roc_auc', cv = cv_split)
# #submit_gbc.fit(data1[data1_x_bin], data1[Target])
# #print('Best Parameters: ', submit_gbc.best_params_) #Best Parameters:  {'learning_rate': 0.25, 'max_depth': 2, 'n_estimators': 50, 'random_state': 0}
# #data_val['Survived'] = submit_gbc.predict(data_val[data1_x_bin])

# #extreme boosting w/full dataset modeling submission score: defaults= 0.73684, tuned= 0.77990
# #submit_xgb = XGBClassifier()
# #submit_xgb = model_selection.GridSearchCV(XGBClassifier(), param_grid= {'learning_rate': grid_learn, 'max_depth': [0,2,4,6,8,10], 'n_estimators': grid_n_estimator, 'seed': grid_seed}, scoring = 'roc_auc', cv = cv_split)
# #submit_xgb.fit(data1[data1_x_bin], data1[Target])
# #print('Best Parameters: ', submit_xgb.best_params_) #Best Parameters:  {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 300, 'seed': 0}
# #data_val['Survived'] = submit_xgb.predict(data_val[data1_x_bin])


# #hard voting classifier w/full dataset modeling submission score: defaults= 0.75598, tuned = 0.77990
# #data_val['Survived'] = vote_hard.predict(data_val[data1_x_bin])
# data_val['Survived'] = grid_hard.predict(data_val[data1_x_bin])


# #soft voting classifier w/full dataset modeling submission score: defaults= 0.73684, tuned = 0.74162
# #data_val['Survived'] = vote_soft.predict(data_val[data1_x_bin])
# #data_val['Survived'] = grid_soft.predict(data_val[data1_x_bin])


#submit filesubmit = data_val[['PassengerId','Survived']]
# submit.to_csv("../working/submit.csv", index=False)

# print('Validation Data Distribution: \n', data_val['Survived'].value_counts(normalize = True))
# submit.sample(10)

