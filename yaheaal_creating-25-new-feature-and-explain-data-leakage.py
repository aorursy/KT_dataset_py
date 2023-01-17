import numpy as np

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, KFold



from sklearn.svm import SVC

from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix



from IPython.display import Image

import warnings

warnings.filterwarnings('ignore')
pure_data = pd.read_csv('../input/heart-disease-uci/heart.csv')



# Remember that we saved our data in "pure_data" so we can train the whole data at the end,

# but for now we will split the data to train and test
print(pure_data.duplicated().sum()) # one duplicated row, let's drop it



pure_data.drop_duplicates(inplace=True)

pure_data.duplicated().any()
# let's take some of the data for the end testing



ratio = 0.2                                    # test samples ratio



count = int(ratio * len(pure_data))            # number of test samples (60)

np.random.seed(42)                             # to give the same random number every time for me and you

rnd = np.random.permutation(len(pure_data))    # list of random numbers from 0 to len(data)



test_idx, train_idx = rnd[:count], rnd[count:]

test, data = pure_data.iloc[test_idx], pure_data.iloc[train_idx]

# for now we are not going to touch "test" so we can use it to evaluate our model at the end



print(data.shape, test.shape)
# let's save our data in original_data so we can play with the data while keeping the original data save



original_data = data.copy()
print(data.shape)

data.head()



# we have some categorical features here, for example sex and ca
data.info()



# no missing data
# we will change some features to "object", so we can describe them better



categorical = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']

data_cat = data.loc[:, categorical].astype('O')
data_cat.describe(include='O')



# freq means the frequency of a value, and top reference to this value

# you can see in fbs the freq is 210 and top is 0, means that only 32 values in fbs are equal to 1
data.describe()



# we need to normalize our numerical features
# we will just use svc, there is no specific reason for that, you can use any other algorithm you want



svc_model = SVC()

svc_model.fit(data.drop('target', axis=1), data['target'])
pred = svc_model.predict(data.drop('target', axis=1))



print('acc =', accuracy_score(data['target'], pred))

print('prec =', precision_score(data['target'], pred))

print('reca =', recall_score(data['target'], pred))



# Oh... not so good :\ , with this numbers I don't think we can apply our model in real life

# let's see what we can do
# categorical features



cat_draw = data.loc[:, categorical]
# count of the values in each feature



fig, ax = plt.subplots(3,3, figsize=(15,10))

plt.suptitle("count of the values in each feature", fontsize=20, fontweight='bold')

fig.patch.set_facecolor('silver')



for i,n in enumerate(cat_draw):

    plt.subplot(3,3,i+1)

    sns.countplot(data[n])

    

# males are more than females

# small amount of people got blood sugar > 120 "fbs = 1"

# small amount of people got slope = 0

# very small amount of people got electrocardiographic(restecg) = 2

# very small amount of people got ca = 4

# very small amount of people got thal = 0

# the data is balanced "ratio of 0 to 1 in the target is good"
# correlation between categorical features and target



fig, ax = plt.subplots(3,3, figsize=(15,10))

plt.suptitle("correlation between categorical features and target", fontsize=20, fontweight='bold')

fig.patch.set_facecolor('silver')

ax.flat[-1].set_visible(False) # to remove last plot



for i,n in enumerate(cat_draw.drop('target',axis=1)):

    plt.subplot(3,3,i+1)

    sns.barplot(x=data[n], y=cat_draw['target'])

    

# They are all good except fbs , we may want to drop it, but first let's try to create new features from it ("fbs")
# correlation between categorical features and fbs with target



cat_with_fbs = ['sex', 'cp', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']



fig, ax = plt.subplots(3,3, figsize=(15,10))

plt.suptitle("correlation between categorical features and fbs with target", fontsize=20, fontweight='bold')

fig.patch.set_facecolor('silver')

ax.flat[-1].set_visible(False) # to remove last plot



for i,n in enumerate(cat_with_fbs):

    plt.subplot(3,3,i+1)

    sns.pointplot(hue='target', y=n, x='fbs', data=data, kind='point');

    

# we can see that ca, cp and exang with fbs are well separated so we will add 3 new features
# correlation between numerical features and fbs with target



fbs_1 = data[data.loc[:, 'fbs'] == 1] # data when fbs = 1

fbs_0 = data[data.loc[:, 'fbs'] == 0] # data when fbs = 0

num_with_fbs = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']



fig, ax = plt.subplots(5,2, figsize=(13,20))

plt.suptitle("correlation between numerical features and fbs with target", fontsize=20, fontweight='bold')

fig.patch.set_facecolor('silver')



for i,n in enumerate(num_with_fbs):

    # left side

    plt.subplot(5,2,i*2+1) # 1, 3, 5, 7, 9

    ax = sns.distplot(fbs_1.loc[fbs_1['target']==1, n], label='targe = 1', bins=10, kde=False) # fbs = 1, target = 1

    ax = sns.distplot(fbs_1.loc[fbs_1['target']==0, n], label='targe = 0', bins=10, kde=False) # fbs = 1, target = 0

    ax.legend()

    if i == 0:

        ax.set_title('fbs = 1', fontsize=17)

    

    # right side

    plt.subplot(5,2,i*2+2) # 2, 4, 6, 8, 10

    ax = sns.distplot(fbs_0.loc[fbs_0['target']==1, n], label='targe = 1', bins=10, kde=False) # fbs = 0, target = 1

    ax = sns.distplot(fbs_0.loc[fbs_0['target']==0, n], label='targe = 0', bins=10, kde=False) # fbs = 0, target = 0

    ax.legend()

    if i == 0:

        ax.set_title('fbs = 0', fontsize=17)

        

# maby I will multiply age with fbs later, if you think I can do something else depending on this plots ,please feel free to write it in the comment
# Normalizing the data

# we will use StandardScaler on the data(numerical data only) 



SS = StandardScaler()

data[num_with_fbs] = SS.fit_transform(data[num_with_fbs])
# the equation used in StandardScaler

# remember this equation because we will use it in the feature
# correlation between numerical features and fbs with target



box_plot = num_with_fbs

fig, ax = plt.subplots(2,3, figsize=(15,10))

plt.suptitle("correlation between numerical features and fbs with target", fontsize=20, fontweight='bold')

fig.patch.set_facecolor('silver')

ax.flat[-1].set_visible(False) # to remove last plot



for i,n in enumerate(box_plot):

    plt.subplot(2,3,i+1)

    sns.boxplot(x='fbs', y=n, hue='target', data=data);

    

# we will talk about how to create new features from box plots but for now try to figure that by yourself :)
# the distribution of the numerical features before using cbrt

# numerical features



dist_data = num_with_fbs



fig, ax = plt.subplots(3,2, figsize=(15,10))

plt.suptitle("the distribution of the numerical features before using cbrt", fontsize=20, fontweight='bold')

fig.patch.set_facecolor('silver')

ax.flat[-1].set_visible(False) # to remove last plot



for i,n in enumerate(dist_data):

    plt.subplot(3,2,i+1)

    sns.distplot(data[n])

    

# we can see that we have right skew in chol and oldpeak so we will try to fix it by using cbrt (the 3rd root) .
data['chol'] = np.cbrt(data['chol'])

data['oldpeak'] = np.cbrt(data['oldpeak'])
# the distribution of the numerical features after using cbrt



dist_data = num_with_fbs



fig, ax = plt.subplots(3,2, figsize=(15,10))

plt.suptitle("the distribution of the numerical features after using cbrt", fontsize=20, fontweight='bold')

fig.patch.set_facecolor('silver')

ax.flat[-1].set_visible(False) # to remove last plot



for i,n in enumerate(dist_data):

    plt.subplot(3,2,i+1)

    sns.distplot(data[n])

    

# we will create new features by separating chol and oldpeak to positive and negative values
# pair plot



pair_plot = dist_data + ['target']

sns.pairplot(data[pair_plot], hue='target');



# here we can see that the data are not good seperated, so it will not be good idea to use linear models

# there is positive correlation (not so big) between age and thalach so we may add new feature from multiplying them together
# fbs with "ca" "exange" and "cp"



data['ca_fbs'] = data['ca'] * data['fbs']

data['exang_fbs'] = data['exang'] * data['fbs']

data['cp_fbs'] = data['cp'] * data['fbs']
data['cut_chol'] = [0 if x<0 else 1 for x in data.chol] # zero if the value is negative and one if positive

data['cut_oldpeak'] = [0 if x<0 else 1 for x in data.oldpeak] # zero if the value is negative and one if positive
data['age_treshtbps'] = data['age'] * data['trestbps']  # "positive correlation" bigger the age ,more the blood pressure

data['age_fbs'] = data['age'] * data['fbs']             # blood sugar can be effected by age

data['sex_cp'] = data['sex'] * data['cp']               # I'am not a doctor but I will assume that the chest pain type can be affected by the sex



data['sex_thalach'] = data['sex'] * data['thalach']     # maximum heart rate achieved (thalach) may be effected by sex and age

data['age_thalach'] = data['age'] * data['thalach']     

data['age_sex_thalach'] = data['age'] * data['sex'] * data['thalach']



# here we will just add some more features depending on my sense ....

data['age_slope'] = data['age'] * data['slope']

data['age_cut_oldpeak'] = data['age'] * data['cut_oldpeak']

data['age_cut_chol'] = data['age'] * data['cut_chol']
ca_target = pd.crosstab(data.ca, data.target)

ca_target.plot.bar(stacked=True);

ca_target



# here we can see that when ca is 0 there is more propelty for the target to be 1 

# while when ca is 1,2,3 the target Tends to be 1

# so we will create new feature
thal_target = pd.crosstab(data.thal, data.target)

thal_target.plot.bar(stacked=True);

thal_target



# as we said before, when thal 2 target 1, else targe 0
ca_target = pd.crosstab(data.cp, data.target)

ca_target.plot.bar(stacked=True);

ca_target



# target = 0 when cp = 0 , else target = 1  
data['ca_1_0'] = [1 if x==0 or x==4 else 0 for x in data.ca]

data['thal_1_0'] = [1 if x==2 or x==0 else 0 for x in data.thal]

data['cp_1_0'] = [0 if x == 0 else 1 for x in data.cp]
sns.catplot(x='slope', y='ca', data=data, hue='target', kind='point');



# you can see how target is well separated between slope and ca

# so we will create from them new feature
sns.catplot(x='slope', y='cp', data=data, hue='target', kind='point');



# new feature from slope and cp
sns.catplot(x='cp', y='ca', data=data, hue='target', kind='point');



# here you can see that some points are overlapping with one another

# so we will not use ca with cp to create new feature
sns.catplot(x='exang', y='ca', data=data, hue='target', kind='point');



# new feature .
data['slope_ca'] = data['slope'] * data['ca']

data['slope_cp'] = data['slope'] * data['cp']

data['exang_ca'] = data['exang'] * data['ca']

data['sex_ca'] = data['sex'] * data['ca'] # you can plot sex with ca like we did before to see why we create new feature from them
sns.boxplot(x='cp', y='restecg', data=data, hue='target');



# nothing to do here
sns.boxplot(x='cp', y='thalach', data=data, hue='target');



# when cp=2 we can see that the blue and orange boxes are seperated 

# meaning that when cp=2 and thalach big target is more to be 1 from 0 ,vice versa

# so new feature is coming :)
sns.boxplot(x='slope', y='oldpeak', data=data, hue='target');



# slope=0 boxes well separated
data['cp_thalach'] = data['cp'] * data['thalach']

data['slope_oldpeak'] = data['slope'] * data['oldpeak']

data['slope_thalach'] = data['slope'] * data['thalach']  # try this by yourself
corr = data.corr()

plt.figure(figsize=(25,15))

sns.heatmap(corr, annot=True, cmap='binary');



# so good , most of the features we created got high corr with the target

# you can see thal_1_0 got 0.53 corr with the target, ca_1_0 and cp_1_1 got 0.48 that's better from all the features we started with
# features that got low corr with target and doesn't affect the accuracy will be dropped

# or that have big corr with other features



data.drop(['fbs', 'chol', 'sex','slope', 'cp', 'age_sex_thalach', 'sex_thalach', 'thalach',

           'age_cut_oldpeak', 'oldpeak', 'slope_ca', 'age_fbs', 'age'], axis=1, inplace=True)
corr = data.corr()

plt.figure(figsize=(25,15))

sns.heatmap(corr, annot=True, cmap='binary');



# much better  :)
# here I just tried to multiply slop_cp and cp_1_0 because the correlation between them is 0.8 

# and it works :') the accuracy improved



data['slope_cp_1_0'] = data['slope_cp'] * data['cp_1_0']
print(data.shape)

data.head() 



# data are ready, now let's see if the score changed
svc_model = SVC()

svc_model.fit(data.drop('target', axis=1), data['target'])

pred = svc_model.predict(data.drop('target', axis=1))



print('acc =', accuracy_score(data['target'], pred))

print('prec =', precision_score(data['target'], pred))

print('reca =', recall_score(data['target'], pred))



# and here is the magic :) , you can see how much better the accuracy is after adding new features

# but what if our model is overfitting , let's evaluate it using k-folds
# create a function to evaluate my model on k-folds



def model_evaluation_wrong(data, algorithm):

    cross_pred, cross_reca, cross_acc = [], [], []

    

    X = data.drop('target', axis=1)

    y = data['target']

    

    model = algorithm.fit(X, y)

    

    pred = model.predict(X)

    

    # train score

    print('train_score :')

    print('  precision ==>' ,round(precision_score(y, pred),2))

    print('  recall ==>', round(recall_score(y, pred),2))

    print('  accuracy ==>', round(accuracy_score(y, pred),2))

    print('*'*40)

    

    # wrong way

    folds = KFold(n_splits=10, random_state=42, shuffle=True)

    for train_idx, test_idx in folds.split(data):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        

        cross_model = algorithm.fit(X_train, y_train)

        pred_cross = cross_model.predict(X_test)

        

        cross_pred.append(precision_score(y_test, pred_cross))

        cross_reca.append(recall_score(y_test ,pred_cross))

        cross_acc.append(accuracy_score(y_test ,pred_cross))

        

    # cross score

    print('cross_score :')

    print('  precision ==> mean:{} | min:{} | max:{} | std:{}'.format(np.mean(cross_pred).round(2), np.min(cross_pred).round(2), np.max(cross_pred).round(2), np.std(cross_pred).round(2)))

    print('  recall ==> mean:{} | min:{} | max:{} | std:{}'.format(np.mean(cross_reca).round(2), np.min(cross_reca).round(2), np.max(cross_reca).round(2), np.std(cross_reca).round(2)))

    print('  accuracy ==> mean:{} | min:{} | max:{} | std:{}'.format(np.mean(cross_acc).round(2), np.min(cross_acc).round(2), np.max(cross_acc).round(2), np.std(cross_acc).round(2)))
# the data I am passing to the function here ,is already processed and that's the problem

# i should process the data while i am splitting the folds



model_evaluation_wrong(data, SVC())  
# we will start with building function that prepare our data automatically, so we can prepare data while doing folds



def data_processing(orig_data, fit=True, mean=None, std=None): 

    # here fit is True when we use the function on the train data ,while False on the test

    # and when it's false we need to pass the mean and the std that we got from the training data

    

    num_data = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']   

    

    # Normalize the data

    if fit:

        SS = StandardScaler().fit(orig_data[num_data])

        orig_data[num_data] = SS.transform(orig_data[num_data])

        mean = SS.mean_

        std = SS.scale_

    if not fit:

        orig_data[num_data] = (orig_data[num_data] - mean) / std  # the equation of StandardScales



    # feature engineering

    orig_data['chol'] = np.cbrt(orig_data['chol'])

    orig_data['oldpeak'] = np.cbrt(orig_data['oldpeak'])

    

    orig_data['ca_fbs'] = orig_data['ca'] * orig_data['fbs']

    orig_data['exang_fbs'] = orig_data['exang'] * orig_data['fbs']

    orig_data['cp_fbs'] = orig_data['cp'] * orig_data['fbs']

    

    orig_data['cut_chol'] = [0 if x<0 else 1 for x in orig_data.chol] 

    orig_data['cut_oldpeak'] = [0 if x<0 else 1 for x in orig_data.oldpeak] 



    orig_data['age_treshtbps'] = orig_data['age'] * orig_data['trestbps']

    orig_data['age_fbs'] = orig_data['age'] * orig_data['fbs']

    orig_data['sex_cp'] = orig_data['sex'] * orig_data['cp']

    orig_data['sex_thalach'] = orig_data['sex'] * orig_data['thalach']

    orig_data['age_thalach'] = orig_data['age'] * orig_data['thalach']

    orig_data['age_sex_thalach'] = orig_data['age'] * orig_data['sex'] * orig_data['thalach']

    orig_data['age_slope'] = orig_data['age'] * orig_data['slope']

    orig_data['age_cut_oldpeak'] = orig_data['age'] * orig_data['cut_oldpeak']

    orig_data['age_cut_chol'] = orig_data['age'] * orig_data['cut_chol']    

    

    orig_data['ca_1_0'] = [1 if x==0 or x==4 else 0 for x in orig_data.ca]

    orig_data['thal_1_0'] = [1 if x==2 or x==0 else 0 for x in orig_data.thal] 

    orig_data['cp_1_0'] = [0 if x == 0 else 1 for x in orig_data.cp]



    orig_data['slope_ca'] = orig_data['slope'] * orig_data['ca']

    orig_data['slope_cp'] = orig_data['slope'] * orig_data['cp']

    orig_data['exang_ca'] = orig_data['exang'] * orig_data['ca']

    orig_data['sex_ca'] = orig_data['sex'] * orig_data['ca']

    

    orig_data['cp_thalach'] = orig_data['cp'] * orig_data['thalach']

    orig_data['slope_oldpeak'] = orig_data['slope'] * orig_data['oldpeak']

    orig_data['slope_thalach'] = orig_data['slope'] * orig_data['thalach']

    orig_data['slope_cp_1_0'] = orig_data['slope_cp'] * orig_data['cp_1_0']

    

    # feature selection

    orig_data.drop(['fbs', 'chol', 'sex','slope', 'cp', 'age_sex_thalach', 'sex_thalach', 'thalach',

                    'age_cut_oldpeak', 'oldpeak', 'slope_ca', 'age_fbs', 'age'], axis=1, inplace=True)

    

    if fit:

        return orig_data, mean, std

    if not fit:

        return orig_data
# to make sure that the function is working the right way



orig_data = original_data.copy() # to keep the original data saved from any change

d, m, s = data_processing(orig_data)



print(d.isin(data).all()) # d is equal to data

print('mean = ', m)

print('std = ', s)



# everything is working good
def model_evaluation_right(orig_data, algorithm):

    cross_pred, cross_reca, cross_acc = [], [], []

    data = orig_data.copy()

    data, _, _ = data_processing(data)

    

    X = data.drop('target', axis=1)

    y = data['target']

    

    model = algorithm.fit(X, y)

    

    pred = model.predict(X)

    

    # train score

    print('train_score :')

    print('  precision ==>' ,round(precision_score(y, pred),2))

    print('  recall ==>', round(recall_score(y, pred),2))

    print('  accuracy ==>', round(accuracy_score(y, pred),2))

    print('*'*40)

    

    # right way

    # we split then we apply data_processing function

    folds = KFold(n_splits=10, random_state=42, shuffle=True)

    X_orig = orig_data.drop('target', axis=1)

    y_orig = orig_data['target']

    

    for train_idx, test_idx in folds.split(orig_data):

        X_train, X_test = X_orig.iloc[train_idx], X_orig.iloc[test_idx]

        y_train, y_test = y_orig.iloc[train_idx], y_orig.iloc[test_idx]

        

        X_train, mean, std = data_processing(X_train)

        # print('mean ==>', mean)                    # PRINT THE MEAN

        # print('std ==>', std,'\n')                 # PRINT THE STD

        X_test = data_processing(X_test, fit=False, mean=mean, std=std)



        cross_model = algorithm.fit(X_train, y_train)

        pred_cross = cross_model.predict(X_test)

        

        cross_pred.append(precision_score(y_test, pred_cross))

        cross_reca.append(recall_score(y_test, pred_cross))

        cross_acc.append(accuracy_score(y_test, pred_cross))



    # cross score

    print('cross_score :')

    print('  precision ==> mean:{} | min:{} | max:{} | std:{}'.format(np.mean(cross_pred).round(2), np.min(cross_pred).round(2), np.max(cross_pred).round(2), np.std(cross_pred).round(2)))

    print('  recall ==> mean:{} | min:{} | max:{} | std:{}'.format(np.mean(cross_reca).round(2), np.min(cross_reca).round(2), np.max(cross_reca).round(2), np.std(cross_reca).round(2)))

    print('  accuracy ==> mean:{} | min:{} | max:{} | std:{}'.format(np.mean(cross_acc).round(2), np.min(cross_acc).round(2), np.max(cross_acc).round(2), np.std(cross_acc).round(2)))
model_evaluation_right(original_data, SVC())
# let's evaluate the model on the test data, don't forget that this test data is untouched . 

# first let's train  the model on the original_data



# prepare the training data

# original_data is the data we start with "train"



prepare_train, mean, std = data_processing(original_data)  

prepare_train.head()
# train our model



svc = SVC()

svc.fit(prepare_train.drop('target', axis=1), prepare_train['target'])
# prepare the test data



prepare_test = data_processing(test, fit=False, mean=mean, std=std)

prepare_test.head()
# confusion_matrix on the test data



sns.heatmap(confusion_matrix(prepare_test['target'], svc.predict(prepare_test.drop('target', axis=1))), annot=True, cmap='binary');
# and at the end don't forget to train you model on the whole data

# pure_data is the data we started with

# mean and std will be used with any new data that we need to predict



whole_data_prepare, mean, std = data_processing(pure_data)

svc = SVC()

svc.fit(whole_data_prepare.drop('target', axis=1), whole_data_prepare['target'])