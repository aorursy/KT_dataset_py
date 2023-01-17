# Importing necessary libraries



import pandas as pd                                                  # DataFrames

import numpy as np                                                   # Mathematical operations

import seaborn as sns                                                # Visualisations

import matplotlib as mplt                                            # Visualisations

import matplotlib.pyplot as plt                                      # Visualisations



import scipy.stats as s                                              # Statistics functions

from scipy.stats import norm, pareto, expon, normaltest, chi2        # Statistics functions

import statsmodels.api as sm                                         # Statistics functions

import statsmodels.discrete.discrete_model as log                    # LogisticRegression

from sklearn.linear_model import LogisticRegression                  # LogisticRegression

from sklearn import svm                                              # SVM





from patsy import dmatrices                                          # Matrix with interactions

import random

import warnings

import itertools





# To speed up run of the kernel, we'll hush down all warnings (it's not advisable to do it at the design stage)

warnings.filterwarnings("ignore")





# Setting seed for repropructive purposes

random.seed(10)
df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.shape
df.head()
pd.isna(df).any()
df.describe()
((np.sum(df.loc[df['age']>50, 'target']))/(np.sum(df.loc[df['age']<50, 'target'])))*(np.sum(df['age']<50)/np.sum(df['age']>50))
# Target

df['target'] = df['target']+1

df.loc[df['target']==2, 'target'] = 0



# Thal

df.loc[df['thal']==0, 'thal'] = np.nan
df = df.rename(columns={'cp':'chest_pain', 'trestbps':'blood_pressure', 'fbs':'blood_sugar', 'restecg':'cardio', 'thalach':'heart_rate', 'exang':'ex_angina', 'oldpeak':'ST_depression', 'ca':'vessels_coloured'})

df2 = df.copy()

df3 = df.copy()
# CHEST PAIN

df.loc[df['chest_pain']==0, 'chest_pain'] = 'asymptomatic'

df.loc[df['chest_pain']==1, 'chest_pain'] = 'typical'

df.loc[df['chest_pain']==2, 'chest_pain'] = 'atypical'

df.loc[df['chest_pain']==3, 'chest_pain'] = 'non_anginal'





# CARDIO

df.loc[df['cardio']==0, 'cardio'] = 'left_ventricular_hypertrophy'

df.loc[df['cardio']==1, 'cardio'] = 'normal'

df.loc[df['cardio']==2, 'cardio'] = 'wave_abnormality'





#SLOPE

df.loc[df['slope']==0, 'slope'] = 'downsloping'

df.loc[df['slope']==1, 'slope'] = 'flat'

df.loc[df['slope']==2, 'slope'] = 'upsloping'





#THAL

df.loc[df['thal']==1, 'thal'] = 'fixed_defect'

df.loc[df['thal']==2, 'thal'] = 'normal'

df.loc[df['thal']==3, 'thal'] = 'reversable_defect'
for i in ['chest_pain', 'cardio', 'slope', 'thal']:

    df = df.merge(pd.get_dummies(df[i], prefix=str(i)), right_index=True, left_index=True)

    df2 = df2.merge(pd.get_dummies(df[i], prefix=str(i)), right_index=True, left_index=True)
sns.set(style="whitegrid", palette="pastel", color_codes=True)

sns.heatmap(df2.corr())
sns.pairplot(df3, hue='target')
sns.pairplot(df3, hue='target', x_vars=['age', 'sex', 'heart_rate', 'slope', 'ST_depression', 'target'], y_vars=['age', 'sex', 'heart_rate', 'slope', 'ST_depression', 'target'])
ax = sns.lmplot('age', 'heart_rate', hue='target', col='target',ci=95, data=df, order=1).set(ylabel="Heart rate", xlabel='Age').fig.suptitle("Effects of age on heart rate", fontsize=25, x=0.53, y=1.05, fontstyle='oblique')
sns.boxplot('slope', 'ST_depression', data=df).set(ylabel="St depression", xlabel='Slope')

plt.title("Differences in means of 'St_depression' in relation the slope", y=1.05, fontsize = 16, fontstyle='oblique')

# Calculating mean of age for each level of slope

mean_up = np.mean(df.loc[df['slope']=='upsloping', 'age'])

mean_flat = np.mean(df.loc[df['slope']=='flat', 'age'])

mean_downsloaping = np.mean(df.loc[df['slope']=='downsloping', 'age'])



# Grand mean

grand_mean = np.mean([mean_up, mean_flat, mean_downsloaping])

variab_y = df['age']-grand_mean

diff_blood_pressure = df['blood_pressure'] - np.mean(df['blood_pressure'])

diff_blood_sugar = df['blood_sugar'] - np.mean(df['blood_sugar'])

diff_heart_rate = df['heart_rate'] - np.mean(df['heart_rate'])





X = pd.DataFrame(np.ones(303), columns=['tau']).merge(diff_blood_pressure, left_index=True, right_index=True).merge(diff_blood_sugar, left_index=True, right_index=True).merge(diff_heart_rate, left_index=True, right_index=True)

from statsmodels.graphics.gofplots import qqplot





est = sm.OLS(variab_y, X)

est2 = est.fit()

print(est2.summary())
sns.despine(left=True)

sns.kdeplot(df.loc[df['target']==1,'age' ], bw=1.5, label="target - 1")

sns.kdeplot(df.loc[df['target']==0,'age'], bw=1.5,label="target - 0")

plt.title("Distributions of age by presence of heart disease", y=1.05, fontsize = 16, fontstyle='oblique')
g = sns.catplot(y='age', x='sex', data=df, hue='target', kind='violin', inner="quart", split=True, palette={0: "y", 1: "b"}).set(ylabel="Age", xlabel='Gender', xticklabels=['female', 'male']).fig.suptitle("Importance of sex and age on heart disease", fontsize=20, x=0.53, y=1.05, fontstyle='oblique')
df.groupby(['sex'])['target'].value_counts()
sns.countplot(x='chest_pain', hue='target', data=df, dodge=True).set_xlabel('Chest pain')

plt.title("Heart disease by chest pain", y=1.05, fontsize = 16, fontstyle='oblique')
chi = np.round(s.chi2_contingency(pd.crosstab(df['chest_pain'], df['target']))[1], 4)

print("The p-value for chi_squared test of independence equals to ",chi, '.')
sns.distplot(df.loc[(df['target']==0), 'ST_depression'], kde_kws={'color':'b', 'label':'target - 0', 'lw':5, 'alpha':0.4})

sns.distplot(df.loc[(df['target']==1), 'ST_depression'], kde_kws={'color':'r', 'label':'target - 1', 'lw':5, 'alpha':0.4})

plt.title("Distribution of ST depression", y=1.05, fontsize = 16, fontstyle='oblique')
b = np.arange(1000)*0.02

plt.plot(b, (-0.4974 - np.log(1.0396+b) + 0.5796/(1.0808+2.3184*b+np.square(b))))
random =  np.random.normal(0, 0.25, 303)

df['sqrt_ST_depression'] = np.sqrt(df['ST_depression'])

sns.distplot(df['sqrt_ST_depression'])
means = lambda x: x.mean()

df_cov = df2.copy()

df_cov['thal'] =  np.where(np.isnan(df2['thal']), 2, df2['thal'])

mean = means(df_cov)

cov = np.linalg.inv(np.cov(df_cov.T)) 

machalonobis = np.diag(np.sqrt(np.dot((df_cov-mean) @ cov, (df_cov-mean).T)))



# Assuming that the test statistic follows chi-square distributed with ‘k’ degree of freedom (where k is the number of predictor variables in the model), we choose the critical value to be 0.025

chi2.ppf((1-0.05), df=14)

np.sum(machalonobis>23.11)



df_no_outliers = df2.copy()

df_no_outliers = df_no_outliers[np.isin(machalonobis, np.sort(machalonobis)[:-7])]



-(df_no_outliers.shape[0] - df.shape[0])
random_numbers_train = np.random.choice(len(df2['age']), size=250, replace=False)

random_numbers_test = np.arange(303)[(np.isin(np.arange(303), random_numbers_train))==False]

train = df2.iloc[random_numbers_train, ]

test = df2.iloc[random_numbers_test, ]





# Outliers (test set - 51, train set - 250, validation set - 48)



random_numbers_train_out = np.random.choice(df_no_outliers.index.values, size=250, replace=False)

random_numbers_test_out = df_no_outliers.index.values[(np.isin(df_no_outliers.index.values, random_numbers_train_out))==False]

train_out = df_no_outliers.loc[random_numbers_train_out, ]

test_out = df_no_outliers.loc[random_numbers_test_out, ]

def standardise(train, test):

    train_target = train['target']

    test_target = test['target']

    train.drop(['target'], axis=1, inplace=True)

    test.drop(['target'], axis=1, inplace=True)

    means = lambda x: x.mean()

    std = lambda x: x.std()

    means_train = means(train)

    std_train = std(train)

    standard = lambda x: (x-means_train)/std_train

    train = standard(train)

    test = standard(test)

    train = pd.merge(train, train_target, left_index=True, right_index=True)

    test = pd.merge(test, test_target, left_index=True, right_index=True)

    return train, test



test.groupby(['cardio_wave_abnormality']).size()

train.groupby(['cardio_wave_abnormality']).size()

# There are only 0 instances of 'cardio_wave_abnormality' in the version without "outliers" and lambda expression calculates standard deviation to equal 0, in which case it divides the feature by 0 and generates NANs.

# Therefore, we need to remove this feature from the version without "outliers"

train_out.drop(['cardio_wave_abnormality'], axis=1, inplace=True)

test_out.drop(['cardio_wave_abnormality'], axis=1, inplace=True)



train, test = standardise(train, test)

train_out, test_out = standardise(train_out, test_out)
def negatives_positives_ratio(y, size, desired_lowest_ratio, desired_highest_ratio):

    y_array = np.array(y).reshape((len(y)))

    all_positions = np.arange(len(y))

    positives_position = all_positions[y_array==1]

    negatives_position = all_positions[y_array==0]

    positives_amount = int(np.sum(y==1))

    negatives_amount = int(np.sum(y==0))

    positiv_max_ratio = np.round(positives_amount/size, 2)

    if positives_amount<desired_highest_ratio*size:

        upper_bound = positiv_max_ratio

    else:

        upper_bound = desired_highest_ratio*size

    if 0.25*size<size-negatives_amount:

        lower_bound = np.round((size-negatives_amount)/size, 2)

    else:

        lower_bound =  desired_lowest_ratio*size

    print(lower_bound, upper_bound)

    positives_number_to_choose = np.round(size*np.arange(lower_bound+0.005, upper_bound, 0.005))

    return positives_number_to_choose,positives_position, negatives_position, positives_amount, negatives_amount, y_array, all_positions







def train_val_split(X, y, k, size, desired_lowest_ratio, desired_highest_ratio):

    preds=[]

    labels=[]

    accuracies=[]

    recalls=[]

    precisions=[]

    trainings_Xs=[]

    trainings_ys=[]

    val_Xs=[]

    val_ys=[]

    X = pd.DataFrame(X)

    y = pd.DataFrame(y)

    positives_number_to_choose,positives_position, negatives_position, positives_amount, negatives_amount, y_array, all_positions = negatives_positives_ratio(y, size, desired_lowest_ratio, desired_highest_ratio)

    for i in range(k):

        positives_number = int(np.random.choice(positives_number_to_choose, 1, replace=False))

        negatives_number = size - positives_number

        positions_train = np.concatenate((np.random.choice(positives_position, positives_number, replace=False), np.random.choice(negatives_position, negatives_number, replace=False)))

        position_val = all_positions[np.isin(all_positions, positions_train)==False]

        training_x = X.iloc[positions_train, ]

        training_y = y_array[positions_train]

        val_x = X.iloc[position_val, ]

        val_y = y_array[position_val]

        trainings_Xs.append(training_x)

        trainings_ys.append(training_y)

        val_Xs.append(val_x)

        val_ys.append(val_y)

    return trainings_Xs, trainings_ys, val_Xs, val_ys





def model_training_prediction(trainings_Xs, trainings_ys, val_Xs):

    preds=[]

    for training_x, training_y, val_x in zip(trainings_Xs, trainings_ys, val_Xs):

        model = LogisticRegression(solver='liblinear').fit(training_x, training_y)

        pred = (model.predict_proba(val_x)[:, 1]).ravel()

        preds.append(pred)

    return preds







def scores_reg(preds, labels, k, threshold, lower_bound_IC, upper_bound_IC):

    accuracy_final = {}

    precision_final = {}

    recall_final = {}

    for cutoff in threshold:

        recalls=[]

        accuracies=[]

        precisions=[]

        for counter in range(k):

            pred = preds[counter]

            label = labels[counter]

            pred = np.where(pred>cutoff, 1, 0)

            pred = np.array(pred.ravel())

#           val_y = val_y.values.reshape((np.sum(np.isin(count, a)==False)))

            TP = np.sum(np.logical_and(pred==1, label==1))

            TN = np.sum(np.logical_and(pred==0, label==0))

            FP = np.sum(np.logical_and(pred==1, label==0))

            FN = np.sum(np.logical_and(pred==0, label==1))

            accuracy = (TP + TN)/(TP + TN + FP + FN)

            recall = TP/(TP+FN)

            precision = TP/(TP+FP)

            recalls.append(recall)

            accuracies.append(accuracy)

            precisions.append(precision)

        recall_mean = np.nanmean(recalls)

        accuracy_mean = np.nanmean(accuracies)

        precision_mean = np.nanmean(precisions)

        recall_lower, recall_upper = np.nanquantile(recalls, np.array([lower_bound_IC, upper_bound_IC]))

        precision_lower, precision_upper = np.nanquantile(precisions,np.array([lower_bound_IC, upper_bound_IC]))

        accuracy_lower, accuracy_upper = np.nanquantile(accuracies,np.array([lower_bound_IC, upper_bound_IC]))

        for scores in [accuracy_mean, recall_mean, precision_mean, recall_lower, recall_upper, precision_lower, precision_upper, accuracy_lower, accuracy_upper]:

            scores = np.round(scores, 3)

        accuracy_final[cutoff] = [accuracy_lower,accuracy_mean, accuracy_upper]

        precision_final[cutoff] = [precision_lower, precision_mean, precision_upper]

        recall_final[cutoff] = [recall_lower, recall_mean, recall_upper]

    return accuracy_final, precision_final, recall_final





def K_fold_regr(X, y, k, size, threshold, lower_bound_IC=0.025, upper_bound_IC=0.975,  desired_lowest_ratio=0.25, desired_highest_ratio=0.75):

    trainings_Xs, trainings_ys, val_Xs, val_ys = train_val_split(X, y, k, size, desired_lowest_ratio, desired_highest_ratio)

    preds = model_training_prediction(trainings_Xs, trainings_ys, val_Xs)

    accuracy_final, precision_final, recall_final = scores_reg(preds, val_ys, k, threshold, lower_bound_IC, upper_bound_IC)

    return accuracy_final, precision_final, recall_final
def negatives_positives_ratio(y, size, desired_lowest_ratio, desired_highest_ratio):

    y_array = np.array(y).reshape((len(y)))

    all_positions = np.arange(len(y))

    positives_position = all_positions[y_array==1]

    negatives_position = all_positions[y_array==0]

    positives_amount = int(np.sum(y==1))

    negatives_amount = int(np.sum(y==0))

    positiv_max_ratio = np.round(positives_amount/size, 2)

    if positives_amount<desired_highest_ratio*size:

        upper_bound = positiv_max_ratio

    else:

        upper_bound = desired_highest_ratio*size

    if 0.25*size<size-negatives_amount:

        lower_bound = np.round((size-negatives_amount)/size, 2)

    else:

        lower_bound =  desired_lowest_ratio*size

    print(lower_bound, upper_bound)

    positives_number_to_choose = np.round(size*np.arange(lower_bound+0.005, upper_bound, 0.005))

    return positives_number_to_choose,positives_position, negatives_position, positives_amount, negatives_amount, y_array, all_positions







def train_val_split(X, y, k, size, desired_lowest_ratio, desired_highest_ratio):

    preds=[]

    labels=[]

    accuracies=[]

    recalls=[]

    precisions=[]

    trainings_Xs=[]

    trainings_ys=[]

    val_Xs=[]

    val_ys=[]

    X = pd.DataFrame(X)

    y = pd.DataFrame(y)

    positives_number_to_choose,positives_position, negatives_position, positives_amount, negatives_amount, y_array, all_positions = negatives_positives_ratio(y, size, desired_lowest_ratio, desired_highest_ratio)

    for i in range(k):

# It's possible that the subsample might have deterministically collinear variables, therefore, we include try/except expression to make the function draw another subsample if such a situation occurs 

        positives_number = int(np.random.choice(positives_number_to_choose, 1, replace=False))

        negatives_number = size - positives_number

        positions_train = np.concatenate((np.random.choice(positives_position, positives_number, replace=False), np.random.choice(negatives_position, negatives_number, replace=False)))

        position_val = all_positions[np.isin(all_positions, positions_train)==False]

        training_x = X.iloc[positions_train, ]

        training_y = y_array[positions_train]

        val_x = X.iloc[position_val, ]

        val_y = y_array[position_val]

        trainings_Xs.append(training_x)

        trainings_ys.append(training_y)

        val_Xs.append(val_x)

        val_ys.append(val_y)

    return trainings_Xs, trainings_ys, val_Xs, val_ys



    

def training_model(X, y, k, size,  classifier, param_name1, param_value1, param_name2=0, param_value2=0, param_name3=0, param_value3=0,param_add_name1='coef0', param_add_value1=0, param_add_name2='gamma', param_add_value2='auto', lower_bound_IC=0.025, upper_bound_IC=0.975,  desired_lowest_ratio=0.25, desired_highest_ratio=0.75):

    accuracies_train = []

    recalls_train = []

    precisions_train = []

    accuracies_out = []

    recalls_out = []

    precisions_out = []

    trainings_Xs, trainings_ys, val_Xs, val_ys = train_val_split(X, y, k, size, desired_lowest_ratio, desired_highest_ratio)

    for training_x, training_y, val_x, val_y in zip(trainings_Xs, trainings_ys, val_Xs, val_ys):

        if isinstance(param_name1, str) & isinstance(param_name2, str)==False:

            param = {param_name1:param_value1, param_add_name1:param_add_value1, param_add_name2:param_add_value2}

            model = classifier(**param).fit(training_x, training_y.ravel())

        elif isinstance(param_name1, str) & isinstance(param_name2, str) & isinstance(param_name3, str)==False:

            param = {param_name1:param_value1, param_name2:param_value2, param_add_name1:param_add_value1, param_add_name2:param_add_value2}

            model = classifier(**param).fit(training_x, training_y.ravel())

        elif isinstance(param_name1, str) & isinstance(param_name2, str) & isinstance(param_name3, str):

            param = {param_name1:param_value1, param_name2:param_value2, param_name3:param_value3,  param_add_name1:param_add_value1, param_add_name2:param_add_value2}

            model = classifier(**param).fit(training_x, training_y.ravel())

        else:

            raise ValueError("Parameter's name must be string")

        for error in ['out-of-sample', 'training']:

            if error=='out-of-sample':

                pred = np.array(model.predict(val_x))

                label = np.array(val_y)

            elif error=='training':

                pred = np.array(model.predict(training_x))

                label = training_y

            TP = np.sum(np.logical_and(pred==1, label==1))

            TN = np.sum(np.logical_and(pred==0, label==0))

            FP = np.sum(np.logical_and(pred==1, label==0))

            FN = np.sum(np.logical_and(pred==0, label==1))         

            accuracy = (TP + TN)/(TP + TN + FP + FN)

            recall = TP/(TP+FN)

            precision = TP/(TP+FP)

            if error=='out-of-sample': 

                recalls_out.append(recall)

                accuracies_out.append(accuracy)

                precisions_out.append(precision)

            elif error=='training':

                recalls_train.append(recall)

                accuracies_train.append(accuracy)

                precisions_train.append(precision)

    return recalls_out,  accuracies_out, precisions_out, recalls_train, accuracies_train, precisions_train



def lower_mean_upper(recalls_out, accuracies_out, precisions_out, recalls_train, accuracies_train, precisions_train, lower_bound_IC, upper_bound_IC):

    recall_mean_out = np.nanmean(recalls_out)

    accuracy_mean_out = np.nanmean(accuracies_out)

    precision_mean_out = np.nanmean(precisions_out)

    recall_mean_train = np.nanmean(recalls_train)

    accuracy_mean_train = np.nanmean(accuracies_train)

    precision_mean_train = np.nanmean(precisions_train)

    recall_lower_out, recall_upper_out = np.nanquantile(recalls_out, np.array([lower_bound_IC, upper_bound_IC]))

    precision_lower_out, precision_upper_out = np.nanquantile(precisions_out,np.array([lower_bound_IC, upper_bound_IC]))

    accuracy_lower_out, accuracy_upper_out = np.nanquantile(accuracies_out,np.array([lower_bound_IC, upper_bound_IC]))

    recall_lower_train, recall_upper_train = np.nanquantile(recalls_train, np.array([lower_bound_IC, upper_bound_IC]))

    precision_lower_train, precision_upper_train = np.nanquantile(precisions_train,np.array([lower_bound_IC, upper_bound_IC]))

    accuracy_lower_train, accuracy_upper_train = np.nanquantile(accuracies_train,np.array([lower_bound_IC, upper_bound_IC])) 

    return recall_mean_out, accuracy_mean_out, precision_mean_out, recall_mean_train, accuracy_mean_train,precision_mean_train, recall_lower_out,recall_upper_out, precision_lower_out, precision_upper_out, accuracy_lower_out, accuracy_upper_out, recall_lower_train, recall_upper_train, precision_lower_train, precision_upper_train, accuracy_lower_train, accuracy_upper_train    



    

def scores(X, y, k, size,  classifier, param_name1, param_value1, param_name2, param_value2, param_name3, param_value3, param_add_name1, param_add_value1, param_add_name2, param_add_value2, lower_bound_IC, upper_bound_IC,  desired_lowest_ratio, desired_highest_ratio):

    recalls_out,  accuracies_out, precisions_out, recalls_train, accuracies_train, precisions_train = training_model(X, y, k, size,  classifier, param_name1, param_value1, param_name2, param_value2, param_name3, param_value3, param_add_name1='coef0', param_add_value1=0, param_add_name2='gamma', param_add_value2='auto', lower_bound_IC=0.025, upper_bound_IC=0.975,  desired_lowest_ratio=0.25, desired_highest_ratio=0.75)

    recall_mean_out, accuracy_mean_out, precision_mean_out, recall_mean_train, accuracy_mean_train,precision_mean_train, recall_lower_out,recall_upper_out, precision_lower_out, precision_upper_out, accuracy_lower_out, accuracy_upper_out, recall_lower_train, recall_upper_train, precision_lower_train, precision_upper_train, accuracy_lower_train, accuracy_upper_train = lower_mean_upper(recalls_out, accuracies_out, precisions_out, recalls_train, accuracies_train, precisions_train, lower_bound_IC, upper_bound_IC)

    return recall_mean_out, accuracy_mean_out, precision_mean_out, recall_mean_train, accuracy_mean_train,precision_mean_train, recall_lower_out,recall_upper_out, precision_lower_out, precision_upper_out, accuracy_lower_out, accuracy_upper_out, recall_lower_train, recall_upper_train, precision_lower_train, precision_upper_train, accuracy_lower_train, accuracy_upper_train



def Hypertuning(X, y, k, size, classifier, params1=0, params2=0, params3=0, param_add_name1='coef0', param_add_value1=0, param_add_name2='gamma', param_add_value2='auto', lower_bound_IC=0.25, upper_bound_IC=0.975,  desired_lowest_ratio=0.25, desired_highest_ratio=0.75):

    accuracies_final_dic_out= {}

    recalls_final_dic_out = {}

    precisions_final_dic_out = {}

    accuracies_final_dic_train= {}

    recalls_final_dic_train = {}

    precisions_final_dic_train = {}

    #

    param_name1 = params1[0]

    for w in range(len(params1[1])):

        param_value1 = params1[1][w]

        if params2!=0:

            param_name2 = params2[0]

            accuracies_final_dic_out[param_value1]= {}

            recalls_final_dic_out[param_value1] = {}

            precisions_final_dic_out[param_value1] = {}

            accuracies_final_dic_train[param_value1]= {}

            recalls_final_dic_train[param_value1] = {}

            precisions_final_dic_train[param_value1] = {}

            for ww in range(len(params2[1])):

                param_value2 = params2[1][ww]

                if params3!=0:

                    param_name3 = params3[0]

                    accuracies_final_dic_out[param_value1][param_value2]= {}

                    recalls_final_dic_out[param_value1][param_value2] = {}

                    precisions_final_dic_out[param_value1][param_value2] = {}

                    accuracies_final_dic_train[param_value1][param_value2]= {}

                    recalls_final_dic_train[param_value1][param_value2] = {}

                    precisions_final_dic_train[param_value1][param_value2] = {}

                    for www in range(len(params3[1])):

                        param_value3 = params3[1][www]

                        recall_mean_out, accuracy_mean_out, precision_mean_out, recall_mean_train, accuracy_mean_train,precision_mean_train, recall_lower_out,recall_upper_out, precision_lower_out, precision_upper_out, accuracy_lower_out, accuracy_upper_out, recall_lower_train, recall_upper_train, precision_lower_train, precision_upper_train, accuracy_lower_train, accuracy_upper_train = scores(X, y, k, size,  classifier, param_name1, param_value1, param_name2, param_value2, param_name3, param_value3, param_add_name1, param_add_value1, param_add_name2, param_add_value2, lower_bound_IC, upper_bound_IC,  desired_lowest_ratio, desired_highest_ratio)

                        accuracies_final_dic_out[param_value1][param_value2][param_value3] = [accuracy_lower_out, accuracy_mean_out, accuracy_upper_out]

                        recalls_final_dic_out[param_value1][param_value2][param_value3] = [recall_lower_out, recall_mean_out, recall_upper_out] 

                        precisions_final_dic_out[param_value1][param_value2][param_value3] = [precision_lower_out, precision_mean_out, precision_upper_out]

                        accuracies_final_dic_train[param_value1][param_value2][param_value3] = [accuracy_lower_train, accuracy_mean_train, accuracy_upper_train]

                        recalls_final_dic_train[param_value1][param_value2][param_value3] = [recall_lower_train, recall_mean_train, recall_upper_train] 

                        precisions_final_dic_train[param_value1][param_value2][param_value3] = [precision_lower_train, precision_mean_train, precision_upper_train]

                else:

                    recall_mean_out, accuracy_mean_out, precision_mean_out, recall_mean_train, accuracy_mean_train,precision_mean_train, recall_lower_out,recall_upper_out, precision_lower_out, precision_upper_out, accuracy_lower_out, accuracy_upper_out, recall_lower_train, recall_upper_train, precision_lower_train, precision_upper_train, accuracy_lower_train, accuracy_upper_train= scores(X, y, k, size,  classifier, param_name1, param_value1, param_name2, param_value2, 0, 0, param_add_name1, param_add_value1, param_add_name2, param_add_value2, lower_bound_IC, upper_bound_IC,  desired_lowest_ratio, desired_highest_ratio)

                    print(param_value1, param_value2, accuracy_mean_out, accuracy_mean_train)

                    accuracies_final_dic_out[param_value1][param_value2] = [accuracy_lower_out, accuracy_mean_out, accuracy_upper_out]

                    recalls_final_dic_out[param_value1][param_value2] = [recall_lower_out, recall_mean_out, recall_upper_out] 

                    precisions_final_dic_out[param_value1][param_value2]= [precision_lower_out, precision_mean_out, precision_upper_out]

                    accuracies_final_dic_train[param_value1][param_value2]= [accuracy_lower_train, accuracy_mean_train, accuracy_upper_train]

                    recalls_final_dic_train[param_value1][param_value2] = [recall_lower_train, recall_mean_train, recall_upper_train] 

                    precisions_final_dic_train[param_value1][param_value2] = [precision_lower_train, precision_mean_train, precision_upper_train]

        else:

            recall_mean_out, accuracy_mean_out, precision_mean_out, recall_mean_train, accuracy_mean_train,precision_mean_train, recall_lower_out,recall_upper_out, precision_lower_out, precision_upper_out, accuracy_lower_out, accuracy_upper_out, recall_lower_train, recall_upper_train, precision_lower_train, precision_upper_train, accuracy_lower_train, accuracy_upper_train= scores(X, y, k, size,  classifier, param_name1, param_value1, 0, 0, 0, 0, param_add_name1, param_add_value1, param_add_name2, param_add_value2, lower_bound_IC, upper_bound_IC,  desired_lowest_ratio, desired_highest_ratio)

            accuracies_final_dic_out[param_value1]= [accuracy_lower_out, accuracy_mean_out, accuracy_upper_out]

            recalls_final_dic_out[param_value1] = [recall_lower_out, recall_mean_out, recall_upper_out] 

            precisions_final_dic_out[param_value1] = [precision_lower_out, precision_mean_out, precision_upper_out]

            accuracies_final_dic_train[param_value1] = [accuracy_lower_train, accuracy_mean_train, accuracy_upper_train]

            recalls_final_dic_train[param_value1] = [recall_lower_train, recall_mean_train, recall_upper_train] 

            precisions_final_dic_train[param_value1] = [precision_lower_train, precision_mean_train, precision_upper_train]

    return  recalls_final_dic_out, precisions_final_dic_out, accuracies_final_dic_out,recalls_final_dic_train, precisions_final_dic_train, accuracies_final_dic_train
# First model: full with all variable and their interaction (of course without one variable per class of variables to avoid multicollinearity)

y, X = dmatrices('target ~ age + sex + age:sex + blood_pressure+ chol+blood_sugar+heart_rate+heart_rate:age+vessels_coloured+ex_angina+ chest_pain_non_anginal + chest_pain_typical + chest_pain_asymptomatic + cardio_left_ventricular_hypertrophy+ cardio_wave_abnormality+slope_flat+slope_upsloping+slope_upsloping:ST_depression+slope_flat:ST_depression+ST_depression+thal_fixed_defect+thal_reversable_defect', data=train)

lr = log.Logit(y, X).fit()

print(lr.summary())





# Second model: with selected variables

y2, X2 = dmatrices('target ~ age + sex + blood_pressure+heart_rate+heart_rate:age+vessels_coloured+chest_pain_atypical + chest_pain_non_anginal + chest_pain_typical + cardio_normal+slope_upsloping+slope_flat+slope_upsloping+slope_flat:ST_depression+thal_reversable_defect', data=train)

lr = log.Logit(y2, X2).fit()

print(lr.summary())
# Creating dataset with interactions also for the dataset without "outliers"

y_out, X_out = dmatrices('target ~ age + sex + age:sex + blood_pressure+ chol+blood_sugar+heart_rate+heart_rate:age+vessels_coloured+ex_angina+ chest_pain_non_anginal + chest_pain_typical + chest_pain_asymptomatic + cardio_left_ventricular_hypertrophy+slope_flat+slope_upsloping+slope_upsloping:ST_depression+slope_flat:ST_depression+ST_depression+thal_fixed_defect+thal_reversable_defect', data=train_out)



y2_out, X2_out = dmatrices('target ~ age + sex + blood_pressure+heart_rate+heart_rate:age+vessels_coloured+chest_pain_atypical + chest_pain_non_anginal + chest_pain_typical + cardio_normal+slope_upsloping+slope_flat+slope_upsloping+slope_flat:ST_depression+thal_reversable_defect', data=train_out)





accuracies_reg, recalls_reg, precisions_reg= K_fold_regr(X, y, k=200, size=200, threshold=np.arange(0, 100, 0.5)*0.01)



accuracies_reg2, recalls_reg2, precisions_reg2 = K_fold_regr(X2, y2, k=200, size=200, threshold=np.arange(0, 100, 0.5)*0.01)



accuracies_reg_out, recalls_reg_out, precisions_reg_out = K_fold_regr(X_out, y_out, k=200, size=200, threshold=np.arange(0, 100, 0.5)*0.01)



accuracies_reg2_out, recalls_reg2_out, precisions_reg2_out = K_fold_regr(X2_out, y2_out, k=200, size=200, threshold=np.arange(0, 100, 0.5)*0.01)
# Function to dismantle lower_bound, mean, upper_bound for each threshold

def lower_mean_upper_score(score):

    threshs, vals = zip(*score.items())

    means_ac = []

    lower_bound_ac=[]

    upper_bound_ac=[]

    for i in range(len(threshs)):

        means_ac.append(vals[i][1])

        lower_bound_ac.append(vals[i][0])

        upper_bound_ac.append(vals[i][2])

    return lower_bound_ac, means_ac, upper_bound_ac







lower_bound_ac, means_ac, upper_bound_ac = lower_mean_upper_score(accuracies_reg)

lower_bound_rc , means_rc , upper_bound_rc = lower_mean_upper_score(recalls_reg)

lower_bound_pr, means_pr, upper_bound_pr = lower_mean_upper_score(precisions_reg)

lower_bound_ac2, means_ac2, upper_bound_ac2 = lower_mean_upper_score(accuracies_reg2)

lower_bound_rc2 , means_rc2 , upper_bound_rc2 = lower_mean_upper_score(recalls_reg2)

lower_bound_pr2, means_pr2, upper_bound_pr2 = lower_mean_upper_score(precisions_reg2)

lower_bound_ac_out, means_ac_out, upper_bound_ac_out = lower_mean_upper_score(accuracies_reg_out)

lower_bound_rc_out , means_rc_out , upper_bound_rc_out = lower_mean_upper_score(recalls_reg_out)

lower_bound_pr_out, means_pr_out, upper_bound_pr_out = lower_mean_upper_score(precisions_reg_out)

lower_bound_ac_out2, means_ac_out2, upper_bound_ac_out2 = lower_mean_upper_score(accuracies_reg2_out)

lower_bound_rc_out2 , means_rc_out2 , upper_bound_rc_out2 = lower_mean_upper_score(recalls_reg2_out)

lower_bound_pr_out2, means_pr_out2, upper_bound_pr_out2 = lower_mean_upper_score(precisions_reg2_out)





lower_bound_acs = [lower_bound_ac, lower_bound_ac2, lower_bound_ac_out, lower_bound_ac_out2]

mean_acs = [means_ac, means_ac2, means_ac_out, means_ac_out2]

upper_bounds_acs = [upper_bound_ac, upper_bound_ac2, upper_bound_ac_out, upper_bound_ac_out2]
colors=['#003f5c', '#f95d6a' ,'#665191', '#ffa600', '#d45087', '#f95d6a', '#820401', '#ff7c43', '#f0a58f']

legends = ['full model', 'reduced model', 'full model no-outlier', 'reduced model no-outliers']



fig, ax = plt.subplots(1, 1, figsize=(12,10))

for score_lower, score_mean, score_upper, color, legend in zip(lower_bound_acs, mean_acs, upper_bounds_acs, colors, legends):

    ax.plot(np.arange(0, 100, 0.5)*0.01, np.round(score_mean, 3), label=legend, color=color)

    ax.legend(loc='upper right')

    ax.fill_between(np.arange(0, 100, 0.5)*0.01, np.round(score_lower, 3), np.round(score_upper, 3), alpha=.1, color=color)

ax.set_xlabel('Threshold', size=15)

ax.set_ylabel('Score', size=15)

ax.set_title('The comparison of accuracy between models \n With confidence interval 95%', fontsize=25, x=0.53, y=1, fontfamily='sans-serif', fontweight="bold")
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

ax.plot(np.arange(0, 100, 0.5)*0.01, means_rc, color='#ff7c43', label='Recall')

ax.set_ylabel('Score', size=15)

ax.set_xlabel('Threshold', size=15)

ax.legend(loc='upper right')

ax.fill_between(np.arange(0, 100, 0.5)*0.01, lower_bound_rc, upper_bound_rc, alpha=.3, color='#ff7c43')

ax.plot(np.arange(0, 100, 0.5)*0.01, means_pr, color='#820401', label='Precision')   

ax.legend(loc='upper right')

ax.fill_between(np.arange(0, 100, 0.5)*0.01, lower_bound_pr, upper_bound_pr, alpha=.3, color='#820401')

ax.set_title('The comparison between Recall and Precision \n With confidence interval 95%', fontsize=25, x=0.53, y=1,  fontfamily='sans-serif', fontweight="bold")
def mean_bounds(score):

    means=[]

    lower_bound=[]

    upper_bound=[]

    for i in list(score.keys()):

        lower_bound.append(score[i][0])

        means.append(score[i][1])

        upper_bound.append(score[i][2])

    return means,lower_bound, upper_bound





def graph_auc(xs, axs, upper, lower, mean, row, col, algorithm_name, method, color):

    axs[col, row].plot(xs, mean, color=color)

    axs[col, row].fill_between(np.arange(len(lower)), lower, upper, alpha=.1, color=color)

    axs[col, row].set_title('{} {}'.format(algorithm_name, method), size=25, y=1.02, fontfamily='serif')



                                                                                      

                                                                                      

def graph_pr_rc(xs, axs, upper_recall, lower_recall, upper_precision, lower_precision,  mean_recall, mean_precision, row, col, algorithm_name, method, color):

    axs[col, row].plot(xs, mean_recall, linestyle='-', color=color)

    axs[col, row].plot(xs, mean_precision, linestyle='-.', color=color)

    axs[col, row].fill_between(np.arange(len(lower_recall)), lower_recall, upper_recall, alpha=.1, color=color)

    axs[col, row].fill_between(np.arange(len(lower_precision)), lower_precision, upper_precision, alpha=.1, color=color)

    axs[col, row].set_title('{} {}'.format(algorithm_name, method), size=25, y=1.02, fontfamily='serif')

                                                                              

                                                                                      

                                                                                                     

                                                                                      

def drawing_ac_pr_rc(ac_score_values_out, pr_score_values_out, rc_score_values_out, ac_score_values_train, pr_score_values_train, rc_score_values_train, algorithm_name, additional_param_name1, additional_param_name2, palette_colours=['#003f5c', '#ff7c43', '#665191', '#ffa600', '#d45087', '#f95d6a', '#820401', '#f0a58f']):

    methods = list(ac_score_values_out.keys())

    method = list(ac_score_values_out.keys())[0]

    method2 = list(ac_score_values_out[method].keys())[0]

    if isinstance(ac_score_values_out[method][method2], list):

        dic_levels=2

        xticks = np.arange(0, len(list(ac_score_values_out[method].keys())), 1)

        xtick_labels = [str(i) for i  in list(ac_score_values_out[method].keys())]

    else:

        dic_levels=3

        xticks = np.arange(0, len(list(ac_score_values_out[method][method2].keys())), 1)

        xtick_labels = [str(i) for i  in list(ac_score_values_out[method][method2].keys())]

    fig, axs = plt.subplots(2, len(methods), figsize=(8*len(methods), 20), sharey=True, squeeze=False)

    fig.suptitle('The Precisions/Recall/Accuracy \n in relation to threshold (95% confidence interval)', fontsize=30,y=0.95, fontfamily='sans-serif', fontweight="bold")

    count=0

    for ax in fig.axes:

        plt.sca(ax)

        plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, hspace=0.3)

        plt.ylim(-0.05, 1.05)

        plt.xlabel('Parameter - C',labelpad=0.25, size=15, fontfamily='serif')

        plt.ylabel('Value of the score', labelpad=15, size=15, fontfamily='serif')

        plt.xticks(xticks, labels=xtick_labels, size=12, fontfamily='serif', rotation=45)

        plt.yticks(np.arange(0, 11)*0.1, label=np.round(np.arange(0, 11)*0.1, 1), size=12, fontfamily='serif')

        plt.grid(axis='y', linestyle='-.', alpha=0.7)

    for score in ['accuracy', 'recall/precision']:

        for counter, method in enumerate(methods):

            if score=='accuracy':

                if dic_levels==3:

                    methods2 = list(ac_score_values_out[method].keys())

                    for color_type, method2 in enumerate(methods2):

                        color = palette_colours[color_type]

                        means, lower_bounds, upper_bounds = mean_bounds(ac_score_values_out[method][method2])

                        xs = np.arange(0, len(list(ac_score_values_out[method][method2].keys())))

                        graph_auc(xs, axs, upper_bounds, lower_bounds, means, counter, 0, algorithm_name, additional_param_name1, color)

                    axs[0, counter].legend(['Accuracy for {}; {}: {}'.format(additional_param_name1, additional_param_name2, method2) for method2 in methods2], fontsize='medium')

                elif dic_levels==2:   

                    means_out, lower_bounds_out, upper_bounds_out = mean_bounds(ac_score_values_out[method])

                    means_train, lower_bounds_train, upper_bounds_train = mean_bounds(ac_score_values_train[method])

                    xs = np.arange(0, len(list(ac_score_values_out[method].keys())))

                    graph_auc(xs, axs, upper_bounds_out, lower_bounds_out, means_out, counter, 0, algorithm_name, method, '#003f5c')

                    graph_auc(xs, axs, upper_bounds_train, lower_bounds_train, means_train, counter, 0, algorithm_name, method, '#f95d6a')

                    axs[0, counter].legend(['Accuracy on validation set', 'Accuracy on training set'], fontsize='large')

            elif score=='recall/precision':

                if dic_levels==3:

                    methods2 = list(ac_score_values_out[method].keys())

                    for color_type, method2 in enumerate(methods2):

                        color = palette_colours[color_type]

                        means_pr, lower_bounds_pr, upper_bounds_pr = mean_bounds(pr_score_values_out[method][method2])

                        means_rc, lower_bounds_rc, upper_bounds_rc = mean_bounds(rc_score_values_out[method][method2])

                        xs = np.arange(0, len(list(ac_score_values_out[method][method2].keys())))

                        graph_pr_rc(xs, axs, upper_bounds_rc, lower_bounds_rc, upper_bounds_pr, lower_bounds_pr,  means_rc, means_pr, counter, 1, algorithm_name, additional_param_name1, color)

                    lg=()

                    for c in range(len(methods2)):

                        lg += ('Recall', 'Precision', )

                    methods2_double = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in methods2))

                    axs[1,counter].legend(['{} for {}; {} : {}'.format(l, additional_param_name1,additional_param_name2, method2) for l, method2  in zip(lg, methods2_double)], fontsize='medium')

                elif dic_levels==2:   

                    means_pr, lower_bounds_pr, upper_bounds_pr = mean_bounds(pr_score_values_out[method])

                    means_rc, lower_bounds_rc, upper_bounds_rc = mean_bounds(rc_score_values_out[method])

                    xs = np.arange(0, len(list(ac_score_values_out[method].keys())))

                    graph_pr_rc(xs, axs, upper_bounds_rc, lower_bounds_rc, upper_bounds_pr, lower_bounds_pr,  means_rc, means_pr, counter, 1, algorithm_name, method, '#ff7c43')

                    axs[1, counter].legend(['Recall', 'Precision'],fontsize='large')
X = train[['age', 'blood_pressure', 'chol', 'blood_sugar','heart_rate', 'ex_angina', 'ST_depression',  'vessels_coloured', 'chest_pain_atypical', 'chest_pain_non_anginal', 'chest_pain_typical', 

        'cardio_left_ventricular_hypertrophy', 'cardio_normal', 'cardio_wave_abnormality', 'slope_downsloping', 'slope_flat','slope_upsloping', 'thal_fixed_defect', 'thal_normal','thal_reversable_defect']]





y = train['target']
recalls_final_dic_out, precisions_final_dic_out, accuracies_final_dic_out,recalls_final_dic_train, precisions_final_dic_train, accuracies_final_dic_train = Hypertuning(X, y, k=200, size=200, classifier=svm.SVC, params1=("kernel", ('linear', 'sigmoid', 'rbf')),

                                                            params2=("C", (0.0001,0.0002, 0.001, 0.002, 0.01,0.02,0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 50)), params3=0)
drawing_ac_pr_rc(accuracies_final_dic_out, precisions_final_dic_out, recalls_final_dic_out, accuracies_final_dic_train,precisions_final_dic_train,  recalls_final_dic_train,'svm', 0, 0)
recalls_final_dic_poly_out, precisions_final_dic_poly_out, accuracies_final_dic_poly_out, recalls_final_dic_poly_train, precisions_final_dic_poly_train, accuracies_final_dic_poly_train = Hypertuning(X, y, k=200, size=200, classifier=svm.SVC, params1=("kernel", ['poly']),

                                                            params2=("degree", (1, 2, 3, 4, 5, 6)), params3=("C", (0.0001,0.0002, 0.001, 0.002, 0.01,0.02,0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0)))
drawing_ac_pr_rc(accuracies_final_dic_poly_out, precisions_final_dic_poly_out, recalls_final_dic_poly_out,  accuracies_final_dic_poly_train, precisions_final_dic_poly_train,  recalls_final_dic_poly_train, 'svm','poly','degree')
recalls_final_dic_poly_1_out, precisions_final_dic_poly_1_out, accuracies_final_dic_poly_1_out, recalls_final_dic_poly_1_train, precisions_final_dic_poly_1_train, accuracies_final_dic_poly_1_train = Hypertuning(X, y, k=200, size=200, classifier=svm.SVC, params1=("kernel", ['poly']),

                                                            params3=("C", (0.0001,0.0002, 0.001, 0.002, 0.01,0.02,0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 50)), params2=('coef0', (0, 0.5, 1, 3, 5, 20, 30)), param_add_name1="degree", param_add_value1=1)
drawing_ac_pr_rc(accuracies_final_dic_poly_1_out, precisions_final_dic_poly_1_out, recalls_final_dic_poly_1_out, accuracies_final_dic_poly_1_train, precisions_final_dic_poly_1_train, recalls_final_dic_poly_1_train, 'Svm', 'linear', 'coeff')
recalls_final_dic_rbf_out, precisions_final_dic_rbf_out, accuracies_final_dic_rbf_out, recalls_final_dic_rbf_train, precisions_final_dic_rbf_train, accuracies_final_dic_rbf_train = Hypertuning(X, y, 200, 200, svm.SVC, ("kernel", ['sigmoid']),

                                                            ('coef0', (0, 0.5, 1, 3, 5, 20, 30)), ("C", (0.0001,0.0002, 0.001, 0.002, 0.01,0.02, 0.1, 0.2, 0.5,1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0)), 'gamma', 'auto')
drawing_ac_pr_rc(accuracies_final_dic_rbf_out, precisions_final_dic_rbf_out, recalls_final_dic_rbf_out, accuracies_final_dic_rbf_out, precisions_final_dic_rbf_out, recalls_final_dic_rbf_out,  'Svm', 'sigmoid', 'gamma')
recalls_final_dic_rbf_gamma_out, precisions_final_dic_rbf_gamma_out, accuracies_final_dic_rbf_gamma_out, recalls_final_dic_rbf_gamma_train, precisions_final_dic_rbf_gamma_train, accuracies_final_dic_rbf_gamma_train= Hypertuning(X, y, k=200, size=200, classifier=svm.SVC, params1=("kernel", ['rbf']),

                                                            params3=("C", (0.0001,0.0002, 0.001, 0.002, 0.01,0.02,0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 50)), params2=('gamma', (0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 1, 2)), param_add_name1="coef0", param_add_value1=0)
drawing_ac_pr_rc(accuracies_final_dic_rbf_gamma_out, precisions_final_dic_rbf_gamma_out, recalls_final_dic_rbf_gamma_out, accuracies_final_dic_rbf_gamma_out, precisions_final_dic_rbf_gamma_out, recalls_final_dic_rbf_gamma_out,  'Svm', 'rbf', 'gamma')
train_y = train['target']

train_x =train.drop(['target'], axis=1)

test_y = test['target']

test_x = test.drop(['target'], axis=1)



#

train['thal'] = np.where(np.isnan(train['thal']), np.nanmean(train['thal']), train['thal'])

sv = svm.SVC(C=0.01, kernel='linear').fit(train_x, train_y)

lr = LogisticRegression().fit(train_x, train_y)

pred_sv_train = sv.predict(train_x)

pred_sv_test = sv.predict(test_x)



accuracy_train_sv = np.sum(pred_sv_train==train_y)/len(pred_sv_train)

pred_lr = lr.predict(test_x)



accuracy_sv = np.sum(pred_sv_test==np.array(test_y))/len(pred_sv_test)

accuracy_lr = np.sum(pred_lr==np.array(test_y))/len(pred_sv_test)
accuracy_sv
accuracy_lr