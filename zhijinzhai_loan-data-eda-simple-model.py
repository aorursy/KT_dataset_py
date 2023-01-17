import os

import numpy as np

import pandas as pd

import datetime 

import seaborn as sns

sns.set_style("dark")



import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
def my_read_file(filename):

    df = pd.read_csv(filename)

    print("{}: Reading {}.".format(now(), filename))

    print("{}: The data contains {} observations with {} columns".format(now(), df.shape[0], df.shape[1]))

    return df



def now():

    tmp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return tmp



# Self-defined function to read dataframe and find the missing data on the columns and # of missing

def checking_na(df):

    try:

        if (isinstance(df, pd.DataFrame)):

            df_na_bool = pd.concat([df.isnull().any(), df.isnull().sum(), (df.isnull().sum()/df.shape[0])*100],

                                   axis=1, keys=['df_bool', 'df_amt', 'missing_ratio_percent'])

            df_na_bool = df_na_bool.loc[df_na_bool['df_bool'] == True]

            return df_na_bool

        else:

            print("{}: The input is not panda DataFrame".format(now()))



    except (UnboundLocalError, RuntimeError):

        print("{}: Something is wrong".format(now()))



loan_data = my_read_file("../input/Loan payments data.csv")

print("\n\n")

print(checking_na(loan_data))
loan_data.head(2)
print(loan_data.loan_status.unique())



fig = plt.figure(figsize=(5,5))

ax = sns.countplot(loan_data.loan_status)

ax.set_title("Count of Loan Status")

for p in ax.patches:

    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))

plt.show()
fig, axs = plt.subplots(1, 2, figsize=(16, 5))

sns.boxplot(x='loan_status', y='Principal', data=loan_data, hue='loan_status', ax=axs[0])

sns.distplot(loan_data.Principal, bins=range(300, 1000, 100), ax=axs[1], kde=True)

plt.show();
print(loan_data[['loan_status', 'Principal', 'Loan_ID']].groupby(['loan_status', 'Principal']).agg(['count']))
fig, axs = plt.subplots(1, 2, figsize=(16,5))

sns.countplot(loan_data.terms, ax=axs[0])

axs[0].set_title("Count of Terms of loan")

for p in axs[0].patches:

    axs[0].annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))



sns.countplot(x='terms', hue='loan_status', data=loan_data, ax=axs[1])

axs[1].set_title("Term count breakdown by loan_status")

for t in axs[1].patches:

    if (np.isnan(float(t.get_height()))):

        axs[1].annotate(0, (t.get_x(), 0))

    else:

        axs[1].annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))



axs[1].legend(loc='upper left')

plt.show();
fig = plt.figure(figsize=(10,5))

ax = sns.countplot(x='effective_date', hue='loan_status', data=loan_data)

ax.set_title('Loan date')

for t in ax.patches:

    if (np.isnan(float(t.get_height()))):

        ax.annotate(0, (t.get_x(), 0))

    else:

        ax.annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))

plt.show();



# Note: if we think that the day of week, or month has the significant factor to the loan status

# Below is the function which we can use to extract the year, month, or day:

# pd.DatetimeIndex(loan_data.effective_date).year

# pd.DatetimeIndex(loan_data.effective_date).month

# pd.DatetimeIndex(loan_data.effective_date).day
loan_data['paid_off_date'] = pd.DatetimeIndex(loan_data.paid_off_time).normalize()

fig = plt.figure(figsize=(16, 6))

ax = sns.countplot(x='paid_off_date', data=loan_data.loc[loan_data.loan_status.isin(['COLLECTION_PAIDOFF', 'PAIDOFF'])] , hue='loan_status')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

for t in ax.patches:

    if (np.isnan(float(t.get_height()))):

        ax.annotate(0, (t.get_x(), 0))

    else:

        ax.annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))



ax.legend(loc='upper right')

plt.show();
# Compute the day to pay-off the loan

loan_data['day_to_pay'] = (pd.DatetimeIndex(loan_data.paid_off_time).normalize() - pd.DatetimeIndex(loan_data.effective_date).normalize()) / np.timedelta64(1, 'D')



fig = plt.figure(figsize=(15, 5))

ax = sns.countplot(x='day_to_pay', hue='terms', data=loan_data)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

for t in ax.patches:

    if (np.isnan(float(t.get_height()))):

        ax.annotate('', (t.get_x(), 0))

    else:

        ax.annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))

        

plt.show();
fig = plt.figure(figsize=(15, 5))

ax = sns.countplot(x='day_to_pay', hue='terms', data=loan_data.loc[loan_data.loan_status == 'PAIDOFF'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

for t in ax.patches:

    if (np.isnan(float(t.get_height()))):

        ax.annotate('', (t.get_x(), 0))

    else:

        ax.annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))

        

plt.show();
tmp = loan_data.loc[(loan_data.day_to_pay > 30) & (loan_data.loan_status == 'PAIDOFF')]

print("{}: Incorrect status: {} observations".format(now(), tmp.shape[0]))

print(tmp[['loan_status', 'terms', 'effective_date', 'due_date', 'paid_off_time']])
fig, axs = plt.subplots(3, 2, figsize=(16, 15))

sns.distplot(loan_data.age, ax=axs[0][0])

axs[0][0].set_title("Total age distribution across dataset")

sns.boxplot(x='loan_status', y='age', data=loan_data, ax=axs[0][1])

axs[0][1].set_title("Age distribution by loan status")

sns.countplot(x='education', data=loan_data, ax=axs[1][0])

axs[1][0].set_title("Education count")

for t in axs[1][0].patches:

    if (np.isnan(float(t.get_height()))):

        axs[1][0].annotate('', (t.get_x(), 0))

    else:

        axs[1][0].annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))



sns.countplot(x='education', data=loan_data, hue='loan_status', ax=axs[1][1])

axs[1][1].set_title("Education by loan status")

for t in axs[1][1].patches:

    if (np.isnan(float(t.get_height()))):

        axs[1][1].annotate('', (t.get_x(), 0))

    else:

        axs[1][1].annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))



axs[1][1].legend(loc='upper right')

sns.countplot(x='Gender', data=loan_data, ax=axs[2][0])

axs[2][0].set_title("# of Gender")

for t in axs[2][0].patches:

    if (np.isnan(float(t.get_height()))):

        axs[2][0].annotate('', (t.get_x(), 0))

    else:

        axs[2][0].annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))



sns.countplot(x='Gender', data=loan_data, hue='education', ax=axs[2][1])

axs[2][1].set_title("Education of the gender")

for t in axs[2][1].patches:

    if (np.isnan(float(t.get_height()))):

        axs[2][1].annotate('', (t.get_x(), 0))

    else:

        axs[2][1].annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))



plt.show();
# Quick view on education + gender => impact to loan_status

pd.crosstab(loan_data.loan_status, loan_data.Gender + "_" + loan_data.education, margins=True)
pd.crosstab(loan_data.loan_status, loan_data.Gender + "_" + loan_data.education, margins=True, normalize='all')
pd.crosstab(loan_data.loan_status, loan_data.Gender + "_" + loan_data.education, margins=True, normalize='index')
pd.crosstab(loan_data.loan_status, loan_data.Gender + "_" + loan_data.education, margins=True, normalize='columns')
loan_data.loc[(loan_data.loan_status == 'PAIDOFF') & (loan_data.day_to_pay > 30), 'loan_status'] = 'COLLECTION_PAIDOFF'
status_map = {"PAIDOFF": 1, "COLLECTION": 2, "COLLECTION_PAIDOFF": 2 }

loan_data['loan_status_trgt'] = loan_data['loan_status'].map(status_map)



fig, axs = plt.subplots(1, 2, figsize=(15, 5))

sns.countplot(x='loan_status', data=loan_data, ax=axs[0])

axs[0].set_title("Count using original target labels")

for t in axs[0].patches:

    if (np.isnan(float(t.get_height()))):

        axs[0].annotate('', (t.get_x(), 0))

    else:

        axs[0].annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))



sns.countplot(x='loan_status_trgt', data=loan_data, ax=axs[1])

axs[1].set_title("Count using new target labels")

for t in axs[1].patches:

    if (np.isnan(float(t.get_height()))):

        axs[1].annotate('', (t.get_x(), 0))

    else:

        axs[1].annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))

    

plt.show();
dummies = pd.get_dummies(loan_data['education']).rename(columns=lambda x: 'is_' + str(x))

loan_data = pd.concat([loan_data, dummies], axis=1)

loan_data = loan_data.drop(['education'],  axis=1)



dummies = pd.get_dummies(loan_data['Gender']).rename(columns=lambda x: 'is_' + str(x))

loan_data = pd.concat([loan_data, dummies], axis=1)

loan_data = loan_data.drop(['Gender'], axis=1)



loan_data = loan_data.drop(['Loan_ID', 'loan_status', 'effective_date', 'due_date', 'paid_off_time', 'past_due_days', 'paid_off_date', 'day_to_pay'], axis=1)
dummy_var = ['is_female', 'is_Master or Above']

loan_data = loan_data.drop(dummy_var, axis = 1)



print(loan_data.head(2))
X = loan_data.drop(['loan_status_trgt'], axis=1)

y = loan_data.loan_status_trgt
# ML library



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm



### Create function to evaluate the score of each classification model

def eval_model_classifier(model, data, target, split_ratio):

    trainX, testX, trainY, testY = train_test_split(data, target, train_size=split_ratio, random_state=0)

    model.fit(trainX, trainY)    

    return model.score(testX,testY)
### 1st round: RandomForestClassification



# Initialise values

num_estimators_array = np.array([1,5,10,50,100,200,500]) 

num_smpl = 5 # Test run the model according to samples_number

num_grid = len(num_estimators_array)

score_array_mu = np.zeros(num_grid) # Keep mean

score_array_sigma = np.zeros(num_grid) # Keep Standard deviation 

j=0



print("{}: RandomForestClassification Starts!".format(now()))

for n_estimators in num_estimators_array:

    score_array = np.zeros(num_smpl) # Initialize

    for i in range(0,num_smpl):

        rf_class = RandomForestClassifier(n_estimators = n_estimators, n_jobs=1, criterion="gini")

        score_array[i] = eval_model_classifier(rf_class, X, y, 0.8)

        print("{}: Try {} with n_estimators = {} and score = {}".format(now(), i, n_estimators, score_array[i]))

    score_array_mu[j], score_array_sigma[j] = np.mean(score_array), np.std(score_array)

    j=j+1



print("{}: RandomForestClassification Done!".format(now()))
fig = plt.figure(figsize=(7,3))

plt.errorbar(num_estimators_array, score_array_mu, yerr=score_array_sigma, fmt='k.-')

plt.xscale("log")

plt.xlabel("number of estimators",size = 16)

plt.ylabel("accuracy",size = 16)

plt.xlim(0.9,600)

plt.ylim(0.3,0.8)

plt.title("Random Forest Classifier", size = 18)

plt.grid(which="both")

plt.show();
C_array = np.array([0.5, 0.1, 1, 5, 10])

score_array = np.zeros(len(C_array))

i=0

for C_val in C_array:

    svc_class = svm.SVC(kernel='linear', random_state=1, C = C_val)

    score_array[i] = eval_model_classifier(svc_class, X, y, 0.8)

    i=i+1



score_mu, score_sigma = np.mean(score_array), np.std(score_array)



fig = plt.figure(figsize=(7,3))

plt.errorbar(C_array, score_array, yerr=score_sigma, fmt='k.-')

plt.xlabel("C assignment",size = 16)

plt.ylabel("accuracy",size = 16)

plt.title("SVM Classifier (Linear)", size = 18)

plt.grid(which="both")

plt.show();
# Note: 

# Gamma: Kernel coefficient - the higher, it will try to exact fit to the training data, hence, can cause overfitting



gamma_array = np.array([0.001, 0.01, 0.1, 1, 10])

score_array = np.zeros(len(gamma_array))

score_mu = np.zeros(len(gamma_array))

score_sigma = np.zeros(len(gamma_array))

i=0

for gamma_val in gamma_array:

    svc_class = svm.SVC(kernel='rbf', random_state=1, gamma = gamma_val)

    score_array[i] = eval_model_classifier(svc_class, X, y, 0.8)

    score_mu[i], score_sigma[i] = np.mean(score_array[i]), np.std(score_array[i])

    i=i+1





fig = plt.figure(figsize=(10,5))

plt.errorbar(gamma_array, score_mu, yerr=score_sigma, fmt='k.-')

plt.xscale('log')

plt.xlabel("Gamma",size = 16)

plt.ylabel("accuracy",size = 16)

plt.title("SVM Classifier (RBF)", size = 18)

plt.grid(which="both")

plt.show();
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.optimizers import SGD



# Change to np.array type

new_x = np.array(X)

new_y = np.array(y)



# fix random seed for reproducibility

np.random.seed(1234)



model = Sequential()

model.add(Dense(64, input_dim=7, init='uniform', activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(new_x, new_y, epochs=150, batch_size=20)

scores = model.evaluate(new_x, new_y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))