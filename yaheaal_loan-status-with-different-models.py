# Importing some important librarys



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
df.shape
df.head()



# We got some categorical data, and it's a binary classification (Yes, NO)
df.info()



# We have missing data , we will handle them as we go
# Describe the numerical data



df.describe()
# we will change the type of Credit_History to object becaues we can see that it is 1 or 0



df['Credit_History'] = df['Credit_History'].astype('O')
# describe categorical data ("objec")



df.describe(include='O')
# we will drop ID because it's not important for our model and it will just mislead the model



df.drop('Loan_ID', axis=1, inplace=True)
df.duplicated().any()



# we got no duplicated rows
# let's look at the target percentage



plt.figure(figsize=(8,6))

sns.countplot(df['Loan_Status']);



print('The percentage of Y class : %.2f' % (df['Loan_Status'].value_counts()[0] / len(df)))

print('The percentage of N class : %.2f' % (df['Loan_Status'].value_counts()[1] / len(df)))



# We can consider it as imbalanced data, but for now i will not
df.columns
df.head(1)
# Credit_History



grid = sns.FacetGrid(df,col='Loan_Status', size=3.2, aspect=1.6)

grid.map(sns.countplot, 'Credit_History');



# we didn't give a loan for most people who got Credit History = 0

# but we did give a loan for most of people who got Credit History = 1

# so we can say if you got Credit History = 1 , you will have better chance to get a loan



# important feature
# Gender



grid = sns.FacetGrid(df,col='Loan_Status', size=3.2, aspect=1.6)

grid.map(sns.countplot, 'Gender');



# most males got loan and most females got one too so (No pattern)



# i think it's not so important feature, we will see later
# Married

plt.figure(figsize=(15,5))

sns.countplot(x='Married', hue='Loan_Status', data=df);



# most people who get married did get a loan

# if you'r married then you have better chance to get a loan :)

# good feature
# Dependents



plt.figure(figsize=(15,5))

sns.countplot(x='Dependents', hue='Loan_Status', data=df);



# first if Dependents = 0 , we got higher chance to get a loan ((very hight chance))

# good feature
# Education



grid = sns.FacetGrid(df,col='Loan_Status', size=3.2, aspect=1.6)

grid.map(sns.countplot, 'Education');



# If you are graduated or not, you will get almost the same chance to get a loan (No pattern)

# Here you can see that most people did graduated, and most of them got a loan

# on the other hand, most of people who did't graduate also got a loan, but with less percentage from people who graduated



# not important feature
# Self_Employed



grid = sns.FacetGrid(df,col='Loan_Status', size=3.2, aspect=1.6)

grid.map(sns.countplot, 'Self_Employed');



# No pattern (same as Education)
# Property_Area



plt.figure(figsize=(15,5))

sns.countplot(x='Property_Area', hue='Loan_Status', data=df);



# We can say, Semiurban Property_Area got more than 50% chance to get a loan



# good feature
# ApplicantIncome



plt.scatter(df['ApplicantIncome'], df['Loan_Status']);



# No pattern
# the numerical data



df.groupby('Loan_Status').median() # median because Not affected with outliers



# we can see that when we got low median in CoapplicantInocme we got Loan_Status = N



# CoapplicantInocme is a good feature
df.isnull().sum().sort_values(ascending=False)
# We will separate the numerical columns from the categorical



cat_data = []

num_data = []



for i,c in enumerate(df.dtypes):

    if c == object:

        cat_data.append(df.iloc[:, i])

    else :

        num_data.append(df.iloc[:, i])
cat_data = pd.DataFrame(cat_data).transpose()

num_data = pd.DataFrame(num_data).transpose()
cat_data.head()
num_data.head()
# cat_data

# If you want to fill every column with its own most frequent value you can use



cat_data = cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))

cat_data.isnull().sum().any() # no more missing data 
# num_data

# fill every missing value with their previous value in the same column



num_data.fillna(method='bfill', inplace=True)

num_data.isnull().sum().any() # no more missing data 
from sklearn.preprocessing import LabelEncoder  

le = LabelEncoder()

cat_data.head()
# transform the target column



target_values = {'Y': 0 , 'N' : 1}



target = cat_data['Loan_Status']

cat_data.drop('Loan_Status', axis=1, inplace=True)



target = target.map(target_values)
# transform other columns



for i in cat_data:

    cat_data[i] = le.fit_transform(cat_data[i])
target.head()
cat_data.head()
df = pd.concat([cat_data, num_data, target], axis=1)
df.head()
X = pd.concat([cat_data, num_data], axis=1)

y = target 
# we will use StratifiedShuffleSplit to split the data Taking into consideration that we will get the same ratio on the target column



from sklearn.model_selection import StratifiedShuffleSplit



sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train, test in sss.split(X, y):

    X_train, X_test = X.iloc[train], X.iloc[test]

    y_train, y_test = y.iloc[train], y.iloc[test]

    

print('X_train shape', X_train.shape)

print('y_train shape', y_train.shape)

print('X_test shape', X_test.shape)

print('y_test shape', y_test.shape)



# almost same ratio

print('\nratio of target in y_train :',y_train.value_counts().values/ len(y_train))

print('ratio of target in y_test :',y_test.value_counts().values/ len(y_test))

print('ratio of target in original_data :',df['Loan_Status'].value_counts().values/ len(df))
# we will use 4 different models for training



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier



models = {

    'LogisticRegression': LogisticRegression(random_state=42),

    'KNeighborsClassifier': KNeighborsClassifier(),

    'SVC': SVC(random_state=42),

    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=1, random_state=42)

}
# loss



from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score



def loss(y_true, y_pred, retu=False):

    pre = precision_score(y_true, y_pred)

    rec = recall_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    loss = log_loss(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)

    

    if retu:

        return pre, rec, f1, loss, acc

    else:

        print('  pre: %.3f\n  rec: %.3f\n  f1: %.3f\n  loss: %.3f\n  acc: %.3f' % (pre, rec, f1, loss, acc))
# train_eval_train



def train_eval_train(models, X, y):

    for name, model in models.items():

        print(name,':')

        model.fit(X, y)

        loss(y, model.predict(X))

        print('-'*30)

        

train_eval_train(models, X_train, y_train)



# we can see that best model is LogisticRegression at least for now, SVC is just memorizing the data so it is overfitting .
X_train.shape
# train_eval_cross

# in the next cell i will be explaining this function



from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)



def train_eval_cross(models, X, y, folds):

    # we will change X & y to dataframe because we will use iloc (iloc don't work on numpy array)

    X = pd.DataFrame(X) 

    y = pd.DataFrame(y)

    idx = [' pre', ' rec', ' f1', ' loss', ' acc']

    for name, model in models.items():

        ls = []

        print(name,':')



        for train, test in folds.split(X, y):

            model.fit(X.iloc[train], y.iloc[train]) 

            y_pred = model.predict(X.iloc[test]) 

            ls.append(loss(y.iloc[test], y_pred, retu=True))

        print(pd.DataFrame(np.array(ls).mean(axis=0), index=idx)[0])  #[0] because we don't want to show the name of the column

        print('-'*30)

        

train_eval_cross(models, X_train, y_train, skf)



# ohhh, as i said SVC is just memorizing the data, and you can see that here DecisionTreeClassifier is better than LogisticRegression 
# some explanation of the above function



x = []

idx = [' pre', ' rec', ' f1', ' loss', ' acc']



# we will use one model

log = LogisticRegression()



for train, test in skf.split(X_train, y_train):

    log.fit(X_train.iloc[train], y_train.iloc[train])

    ls = loss(y_train.iloc[test], log.predict(X_train.iloc[test]), retu=True)

    x.append(ls)

    

# thats what we get

pd.DataFrame(x, columns=idx)



# (column 0 represent the precision_score of the 10 folds)

# (row 0 represent the (pre, rec, f1, loss, acc) for the first fold)

# then we should find the mean of every column

# pd.DataFrame(x, columns=idx).mean(axis=0)
# ooh, we got it right for most of the features, as you can see we've say at the first of the kernel ,

# that Credit_Histroy and Married etc, are good features, actually Credit_Histroy is the best .



data_corr = pd.concat([X_train, y_train], axis=1)

corr = data_corr.corr()

plt.figure(figsize=(10,7))

sns.heatmap(corr, annot=True);



# here we got 58% similarity between LoanAmount & ApplicantIncome 

# and that may be bad for our model so we will see what we can do
X_train.head()
# I will try to make some operations on some features, here I just tried diffrent operations on diffrent features,

# having experience in the field, and having knowledge about the data will also help



X_train['new_col'] = X_train['CoapplicantIncome'] / X_train['ApplicantIncome']  

X_train['new_col_2'] = X_train['LoanAmount'] * X_train['Loan_Amount_Term'] 
data_corr = pd.concat([X_train, y_train], axis=1)

corr = data_corr.corr()

plt.figure(figsize=(10,7))

sns.heatmap(corr, annot=True);



# new_col 0.03 , new_col_2, 0.047

# not that much , but that will help us reduce the number of features
X_train.drop(['CoapplicantIncome', 'ApplicantIncome', 'Loan_Amount_Term', 'LoanAmount'], axis=1, inplace=True)
train_eval_cross(models, X_train, y_train, skf)



# ok, SVC is improving, but LogisticRegression is overfitting

# i wan't change nothing so we can see what will happen as we go
# first lets take a look at the value counts of every label



for i in range(X_train.shape[1]):

    print(X_train.iloc[:,i].value_counts(), end='\n------------------------------------------------\n')
# new_col_2



# we can see we got right_skewed

# we can solve this problem with very simple statistical teqniq , by taking the logarithm of all the values

# because when data is normally distributed that will help improving our model



from scipy.stats import norm



fig, ax = plt.subplots(1,2,figsize=(20,5))



sns.distplot(X_train['new_col_2'], ax=ax[0], fit=norm)

ax[0].set_title('new_col_2 before log')



X_train['new_col_2'] = np.log(X_train['new_col_2'])  # logarithm of all the values



sns.distplot(X_train['new_col_2'], ax=ax[1], fit=norm)

ax[1].set_title('new_col_2 after log');
# now we will evaluate our models, and i will do that continuously ,so i don't need to mention that every time



train_eval_cross(models, X_train, y_train, skf)



# wooow our models improved really good by just doing the previous step .
# new_col



# most of our data is 0 , so we will try to change other values to 1



print('before:')

print(X_train['new_col'].value_counts())



X_train['new_col'] = [x if x==0 else 1 for x in X_train['new_col']]

print('-'*50)

print('\nafter:')

print(X_train['new_col'].value_counts())
train_eval_cross(models, X_train, y_train, skf)



# ok we are improving our models as we go 
for i in range(X_train.shape[1]):

    print(X_train.iloc[:,i].value_counts(), end='\n------------------------------------------------\n')

    

# looks better
# we will use boxplot to detect outliers



sns.boxplot(X_train['new_col_2']);

plt.title('new_col_2 outliers', fontsize=15);

plt.xlabel('');
threshold = 0.1  # this number is hyper parameter , as much as you reduce it, as much as you remove more points

                 # you can just try different values the deafult value is (1.5) it works good for most cases

                 # but be careful, you don't want to try a small number because you may loss some important information from the data .

                 # that's why I was surprised when 0.1 gived me the best result

            

new_col_2_out = X_train['new_col_2']

q25, q75 = np.percentile(new_col_2_out, 25), np.percentile(new_col_2_out, 75) # Q25, Q75

print('Quartile 25: {} , Quartile 75: {}'.format(q25, q75))



iqr = q75 - q25

print('iqr: {}'.format(iqr))



cut = iqr * threshold

lower, upper = q25 - cut, q75 + cut

print('Cut Off: {}'.format(cut))

print('Lower: {}'.format(lower))

print('Upper: {}'.format(upper))



outliers = [x for x in new_col_2_out if x < lower or x > upper]

print('Nubers of Outliers: {}'.format(len(outliers)))

print('outliers:{}'.format(outliers))



data_outliers = pd.concat([X_train, y_train], axis=1)

print('\nlen X_train before dropping the outliers', len(data_outliers))

data_outliers = data_outliers.drop(data_outliers[(data_outliers['new_col_2'] > upper) | (data_outliers['new_col_2'] < lower)].index)



print('len X_train before dropping the outliers', len(data_outliers))
X_train = data_outliers.drop('Loan_Status', axis=1)

y_train = data_outliers['Loan_Status']
sns.boxplot(X_train['new_col_2']);

plt.title('new_col_2 without outliers', fontsize=15);

plt.xlabel('');



# good :)
train_eval_cross(models, X_train, y_train, skf)



# know we got 94.1 for precision & 53.5 for recall
# Self_Employed got really bad corr (-0.00061) , let's try remove it and see what will happen



data_corr = pd.concat([X_train, y_train], axis=1)

corr = data_corr.corr()

plt.figure(figsize=(10,7))

sns.heatmap(corr, annot=True);
#X_train.drop(['Self_Employed'], axis=1, inplace=True)



train_eval_cross(models, X_train, y_train, skf)



# looks like Self_Employed is not important

# KNeighborsClassifier improved



# droping all the features Except for Credit_History actually improved KNeighborsClassifier and didn't change anything in other models

# so you can try it by you self

# but don't forget to do that on testing data too



#X_train.drop(['Self_Employed','Dependents', 'new_col_2', 'Education', 'Gender', 'Property_Area','Married', 'new_col'], axis=1, inplace=True)
data_corr = pd.concat([X_train, y_train], axis=1)

corr = data_corr.corr()

plt.figure(figsize=(10,7))

sns.heatmap(corr, annot=True);
X_test.head()
X_test_new = X_test.copy()
x = []



X_test_new['new_col'] = X_test_new['CoapplicantIncome'] / X_test_new['ApplicantIncome']  

X_test_new['new_col_2'] = X_test_new['LoanAmount'] * X_test_new['Loan_Amount_Term']

X_test_new.drop(['CoapplicantIncome', 'ApplicantIncome', 'Loan_Amount_Term', 'LoanAmount'], axis=1, inplace=True)



X_test_new['new_col_2'] = np.log(X_test_new['new_col_2'])



X_test_new['new_col'] = [x if x==0 else 1 for x in X_test_new['new_col']]



#X_test_new.drop(['Self_Employed'], axis=1, inplace=True)



# drop all the features Except for Credit_History

#X_test_new.drop(['Self_Employed','Dependents', 'new_col_2', 'Education', 'Gender', 'Property_Area','Married', 'new_col'], axis=1, inplace=True)
X_test_new.head()
X_train.head()
for name,model in models.items():

    print(name, end=':\n')

    loss(y_test, model.predict(X_test_new))

    print('-'*40)