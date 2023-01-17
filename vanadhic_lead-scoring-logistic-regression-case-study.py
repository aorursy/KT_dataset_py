import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline



pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Read the dataset Leads.csv



df = pd.read_csv("/kaggle/input/lead-scoring-dataset/Lead Scoring.csv")

df.head()


df.head()
# Take a copy of the original dataset to assign the Lead score to the original rows. 



df_orig = df.copy()
df.describe()
df.shape
#df.info()
# Dividing the dataset into two dataset with Converted = 0 and Converted = 1



df_0=df.loc[df["Converted"]==0]

df_1=df.loc[df["Converted"]==1]
# Calculating Imbalance percentage 

# Since the majority is target0 and minority is target1

print (f'Count of Converted = 0: {len(df_0)} \nCount of Converted = 1: {len(df_1)}')

print (f'Imbalance Ratio is : {round(len(df_0)/len(df_1),2)}')
# Plotting the imbalance Analysis:

sns.set_style('whitegrid')

fig, ax = plt.subplots(figsize = (6,4))

plt.title('Imbalance Analysis',  fontsize=20)

chart = sns.countplot(data=df, x='Converted', palette='muted')

plt.xlabel('Converted', fontsize=18)

plt.ylabel('count', fontsize=18)
# Converting 'Select' values to NaN.

df = df.replace('Select', np.nan)
row1 , column1 = df.shape[0], df.shape[1]



# delete duplicates

df = df.drop_duplicates() 
row2 , column2 = df.shape[0], df.shape[1]



percentRows = round ((row2/row1 * 100), 2)

print (f'Rows retained after Duplicate Deletion: {row2} or {percentRows} percent')
# To find percent of Nan values

# We can define a function to get the missing values and missing percentage for the dataframes.

def missing_data(data):

    count_missing = data.isnull().sum().sort_values(ascending=False)

    percent_missing = (data.isnull().sum() * 100 / len(data)).sort_values(ascending=False)

    missing_value_df = pd.DataFrame({'count_missing': count_missing,

                                 'percent_missing': percent_missing})

    return missing_value_df
#To find percent of Nan values 

missing_data(df).head(20).transpose()
# To check if there are any duplicate values in Prospect ID and Lead Number columns



print (f'Duplicates in Prospect ID - {any(df["Prospect ID"].duplicated())}')

print (f'Duplicates in Lead Number - {any(df["Lead Number"].duplicated())}')
# Dropping the columns as mentioned in the above comment. 

dropFeatures = ['Prospect ID', 'Lead Number']

df.drop(df[dropFeatures], axis=1, inplace=True)
# we will drop the columns having more than 70% NA values.

def drop_columns(data, miss_per):

    cols_to_drop = list(round(100*(data.isnull().sum()/len(data.index)), 2) >= miss_per )

    dropcols = data.loc[:,cols_to_drop].columns

    print (f'Features dropping now: {dropcols}')

    data = data.drop(dropcols, axis=1)

    return data
df = drop_columns(df, 70.0)
#missing_data(df).head(20)
# Analyse the score columns assigned by the sales team to the dataset before dropping them



scoreFeatures = ['Lead Quality', 'Asymmetrique Activity Index', 'Asymmetrique Profile Index' ]



# Count plot for the categorical variables

sns.set(style='ticks',color_codes=True)

colors =['Accent', 'PiYG' , 'RdPu']



plt.figure(figsize = (15,5))

for i in enumerate(scoreFeatures):

    plt.subplot(1, 3, i[0]+1)

    chart = sns.countplot(x = i[1], hue = 'Converted', data = df, palette = colors[i[0]])

    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha='right',)

    plt.tight_layout()
# Analyse the score columns assigned by the sales team to the dataset



fig, axis = plt.subplots(1, 2, figsize = (12,4))

plt1 = sns.distplot(df_0['Asymmetrique Activity Score'], hist=False, kde=True , color='b' , ax = axis[0])

plt1 = sns.distplot(df_1['Asymmetrique Activity Score'], hist=False, kde=True , color='r' , ax = axis[0])

plt2 = sns.distplot(df_0['Asymmetrique Profile Score'], hist=False, kde=True , color='b' , ax = axis[1])

plt2 = sns.distplot(df_1['Asymmetrique Profile Score'], hist=False, kde=True , color='r' , ax = axis[1])

plt.tight_layout()
# Drop the score columns assigned by the sales team to the dataset



df = drop_columns(df, 45.0)
df.columns
# Drop the unwanted features

dropFeatures = ['Tags', 'Last Notable Activity']



df.drop(dropFeatures, axis=1, inplace=True)
# A function to find the constant features. Constant features are those features which have only one distinct value.



def find_constant_features(df):

    constFeatures = []

    for column in list(df.columns):

        if df[column].unique().size < 2:

            constFeatures.append(column)

    return constFeatures



constFeatures = find_constant_features(df)

print(constFeatures)
# Drop the constant features as they will not add value to the analysis



df = df.drop(constFeatures, axis=1)
df.shape
# Look at the number of unique categories in a column

def unique_count(data):

    data_type = data.dtypes

    unique_count = data.nunique()

    

    unique_count_df = pd.DataFrame({'data_type': data_type,

                                 'unique_count': unique_count})

    return unique_count_df
unique_count(df).transpose() # Used transpose so as to avoid using more space. `
# Identify and separate all the Categorical, boolean and numeric features for analysis

targetFeature = []

catFeatures = []

boolFeatures = []

numFeatures = []



for each in df.columns:

    if each in ('Converted'):

        targetFeature.append(each)

    elif df[each].nunique() == 2:  #Features with only 2 unique values as boolean

        boolFeatures.append(each)

    elif df[each].dtype == 'object':

        catFeatures.append(each)

    elif df[each].dtype in ('int64','float64'):

        numFeatures.append(each)

    else:

        numFeatures.append(each)
print (f'The Target Feature is : \n {targetFeature} \n')

print (f'The Boolean Features are : \n {boolFeatures} \n')

print (f'The Categorical Features are : \n {catFeatures} \n')

print (f'The Numeric Features are :\n {numFeatures} \n')
boolFeatures
# Convert the values 'Yes' and 'No' to 1 and 0 in the Binary Features. 

# value_counts is checked each time to ensure the mapping is done only once 

# If mapped multiple times, the values are converted to NaNs



for each in boolFeatures:

    if df[each].value_counts().values.sum() > 0:  # To check if the step was already completed

        df[each] = df[each].map(dict(Yes=1, No=0))

        print (f'Binary mapping is completed for {each}')
# Convert the boolean features to type boolean

df[boolFeatures] = df[boolFeatures].astype('int64')
boolFeatures
df.shape
# Count plot for the Boolean variables

# colors = ['Accent', 'PiYG' , 'RdPu', 'icefire' , 'ocean' , 'gist_earth', 'magma', 'plasma', 'rocket']

colors = ['Accent', 'ocean', 'rocket'] * 3

sns.set(style='ticks',color_codes=True)

plt.figure(figsize = (10,10))

for i, x_var in enumerate(boolFeatures):

    plt.subplot(3, 3, i+1)

    chart = sns.countplot(x = x_var, data = df, hue='Converted', palette=colors[i])

    chart.set_xticklabels(chart.get_xticklabels())

    plt.tight_layout()
# Identify the value counts of the boolean features to confirm if they have only one value



for each in boolFeatures:

    print (df[each].value_counts(dropna=False))
# we can drop the boolean Features with most values as 0 as they all have the value True and do not help in the analysis



dropFeatures = [ 'Do Not Call',

                 'Search',

                 'Newspaper Article',

                 'X Education Forums',

                 'Newspaper',

                 'Digital Advertisement',

                 'Through Recommendations']
# Drop the unwanted features



df.drop(dropFeatures, axis=1, inplace=True)
#To find percent of Nan values 

missing_data(df).head(10)
df.shape
numFeatures
# Analyze the numeric features



sns.set(style='ticks',color_codes=True)

fig = plt.figure(figsize = (15, 15))

g = sns.pairplot(data=df, hue='Converted', vars=numFeatures + targetFeature);
# Frequency Ditribution for Numeric Features

sns.set(style='ticks',color_codes=True)

plt.figure(figsize = (12, 12))

for i, x_var in enumerate(numFeatures):

    plt.subplot(3, 2, i+1)

    sns.distplot(df_0[x_var], hist=False, kde=True , color='b')

    sns.distplot(df_1[x_var], hist=False, kde=True , color='r')

    plt.tight_layout()
df.Converted.dtype
numFeatures
# Box plot to identify the outliers

# Frequency Ditribution for Numeric Features

sns.set(style='ticks',color_codes=True)

colors = ['Accent', 'ocean' , 'RdPu']

plt.figure(figsize = (12, 12))

for i, var in enumerate(numFeatures):

    plt.subplot(3,3,i+1)

    sns.boxplot(x='Converted', y = var, data = df, palette =colors[i])

    plt.tight_layout()
cap_outliers = ['TotalVisits', 'Page Views Per Visit']
# Cap the outliers for the Numeric features at 0.01 and 0.99



for i, var in enumerate(cap_outliers):

    q1 = df[var].quantile(0.01)

    q4 = df[var].quantile(0.99)

    df[var][df[var]<=q1] = q1

    df[var][df[var]>=q4] = q4
# Box plot to visualise numeric features after outlier capping

sns.set(style='ticks',color_codes=True)

colors = ['Accent', 'ocean' , 'RdPu'] # 'icefire' , 'ocean' , 'gist_earth', 'magma', 'prism', 'rocket', 'seismic']

plt.figure(figsize = (12, 12))

for i, var in enumerate(numFeatures):

    plt.subplot(3,3,i+1)

    sns.boxplot(x = 'Converted', y = var, data = df, palette=colors[i])

    plt.tight_layout()
# Impute the missing values for the columns with Mean



df['TotalVisits'].fillna((df['TotalVisits'].mean()), inplace=True)

df['Page Views Per Visit'].fillna((df['Page Views Per Visit'].mean()), inplace=True)
# Correlation Heat map for the numeric features



corrFeatures = numFeatures + targetFeature



sns.set(style='ticks',color_codes=True)

plt.figure(figsize = (6,6))



sns.heatmap(df[corrFeatures].corr(), cmap="YlGnBu", annot=True, square=True)

plt.show()
numFeatures
#To find percent of Nan values 

#missing_data(df).head(10)
# Identify the Unique Counts for the categorical Features



unique_count(df[catFeatures]).transpose() # Used transpose so as to avoid using more space. `
catFeatures
unique_count(df[catFeatures]).sort_values(by = 'unique_count', ascending=False)
catFeatures[:4]

catFeatures[4:]
# Count plot for the categorical variables

sns.set(style='ticks',color_codes=True)

# colors =['Accent', 'PiYG' , 'RdPu', 'icefire' , 'ocean' , 'gist_earth', 'magma', 'prism', 'rocket', 'seismic']

colors =['gist_earth', 'magma', 'ocean', 'rocket'] * 2

plt.figure(figsize = (15,12))

for i, x_var in enumerate(catFeatures[:4]):

    plt.subplot(2, 2, i+1)

    chart = sns.countplot(x = x_var, hue = 'Converted', data = df, palette = colors[i])

    chart.set_xticklabels(chart.get_xticklabels(), fontsize=14, rotation=45, ha='right',)

    plt.xlabel(x_var, fontsize=14)

    plt.ylabel('count', fontsize=14)

    plt.tight_layout()
# Count plot for the categorical variables

sns.set(style='ticks',color_codes=True)

# colors =['Accent', 'PiYG' , 'RdPu', 'icefire' , 'ocean' , 'gist_earth', 'magma', 'prism', 'rocket', 'seismic']

colors =['gist_earth', 'magma', 'ocean', 'rocket'] * 2

plt.figure(figsize = (15,12))

for i, x_var in enumerate(catFeatures[4:]):

    plt.subplot(2, 2, i+1)

    chart = sns.countplot(x = x_var, hue = 'Converted', data = df, palette = colors[i])

    chart.set_xticklabels(chart.get_xticklabels(), fontsize=14, rotation=45, ha='right',)

    plt.xlabel(x_var, fontsize=14)

    plt.ylabel('count', fontsize=14)

    plt.tight_layout()
df.columns
dropFeatures = ['Country', 'What matters most to you in choosing a course']



df.drop(dropFeatures, axis=1, inplace=True)
df.columns
catFeatures = []



for each in df.columns:

    if df[each].dtype == 'object':

        catFeatures.append(each)



catFeatures
df['Lead Source'] = df['Lead Source'].replace(['google'], 'Google')
# Replace all the NaN values for categorical variables

df['City'] = df['City'].replace(np.nan, 'Mumbai')
for each in catFeatures:

    print (f'Value Counts for {each}: \n {df[each].value_counts(dropna=False)} \n')
# Since there are so many categories in the categorical features with less than 2% counts each, we can 

# combine all those categories into one category called 'Others'



for each in catFeatures:

    replaceFeatures = []

    categories = df[each].value_counts()

    list1 = df[each].value_counts().keys().tolist()

    for i, v in enumerate (categories):

        if v <= 200:  ## Anything less than 200

            replaceFeatures.append(list1[i])

    df[each] = df[each].replace(replaceFeatures, 'Others')

    print (f'Categories replaced for column {each} are: \n {replaceFeatures} \n')
#To find percent of Nan values 

# missing_data(df).head(20)
# Replace all the NaN values with 'Missing' for the remaining Categorical variables with NaN in them

nanFeatures = ['Specialization', 'What is your current occupation', 'Lead Source', 'Last Activity']



for each in nanFeatures:

    df[each].replace(np.nan,'Missing', inplace=True)

    print (f'NaNs are converted to "Missing" category for column {each}')
catFeatures
# Count plot for the categorical variables

sns.set(style='ticks',color_codes=True)

plt.figure(figsize = (25, 18))

colors = [ 'RdBu', 'rocket' , 'gist_earth'] * 2

for i, x_var in enumerate(catFeatures):

    plt.subplot(2, 3, i+1)

    chart = sns.countplot(x = x_var, hue = 'Converted', data = df, palette = colors[i])

    chart.set_xticklabels(chart.get_xticklabels(), fontsize=16, rotation=45, ha='right')

    plt.xlabel(x_var, fontsize=16)

    plt.ylabel('count', fontsize=16)

    plt.tight_layout()
#To find percent of Nan values 

missing_data(df).head()
catFeatures
# Getting dummy variables and adding the results to the master dataframe



for each in catFeatures:

    dummy = pd.get_dummies(df[each], drop_first=False, prefix=each)

    df = pd.concat([df,dummy],1)

    print (f'dummy columns are added for the feature {each}')
# Drop the sepcific dummy columns created after the dummy variables are added for these categorical columns



dummydropFeatures = ['Lead Origin_Others', 

                     'City_Others',

                     'Lead Source_Missing',

                     'Specialization_Missing',

                     'What is your current occupation_Missing',

                     'Last Activity_Missing']



df.drop(dummydropFeatures, axis=1, inplace=True )
catFeatures
# Drop the original categorical columns since the dummy variables are added for these categorical columns



df.drop(catFeatures, axis=1, inplace=True )
df.head()
df.columns
from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression

#from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler
# The target variable in y

y = df['Converted']

y.head()
# The feature variables in X



X=df.drop('Converted', axis=1)

X.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, test_size=0.3, random_state=101)
numFeatures
#### Scaling the numerical columns

scaler = MinMaxScaler()



X_train[numFeatures] = scaler.fit_transform(X_train[numFeatures])



X_train.head()
# Build the Logistic Regression Model

logmodel = LogisticRegression()



from sklearn.feature_selection import RFE

rfe = RFE(logmodel, 20)             # running RFE with 20 variables as output

rfe = rfe.fit(X_train, y_train)
# print (rfe.support_)

# list(zip(X_train.columns, rfe.support_, rfe.ranking_))
#list of RFE supported columns

cols = X_train.columns[rfe.support_]

cols
# Defining a function to generate the model by passing the model name and the columns used for the model 



def gen_model(model_no, cols):

    X_train_sm = sm.add_constant(X_train[cols])

    model_no = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

    res = model_no.fit()

    print (res.summary())

    return res
from statsmodels.stats.outliers_influence import variance_inflation_factor



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

def calcVIF(col):

    vif = pd.DataFrame()

    vif['Features'] = X_train[col].columns

    vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return vif
# Generate the first model using the RFE features



logm1 = LogisticRegression()



#Pass the columns to generate the model and print summary

res = gen_model(logm1, cols)



# Check the VIF for the features

calcVIF(cols).head(3)
res
# Dropping the next unwanted variable to pass to the model.

cols = cols.drop('Specialization_Supply Chain Management',1)

logm2 = LogisticRegression()



#Pass the columns to generate the model and print summary

res = gen_model(logm2, cols)



# Check the VIF for the features

calcVIF(cols).head(3)
# Dropping the next unwanted variable to pass to the model.

cols = cols.drop('Specialization_Banking, Investment And Insurance',1)

logm3 = LogisticRegression()



#Pass the columns to generate the model and print summary

res = gen_model(logm3, cols)



# Check the VIF for the features

calcVIF(cols).head(3)
# Dropping the next unwanted variable to pass to the model.

cols = cols.drop('Specialization_Finance Management',1)

logm4 = LogisticRegression()



#Pass the columns to generate the model and print summary

res = gen_model(logm4, cols)



# Check the VIF for the features

calcVIF(cols).head(3)
# Dropping the next unwanted variable to pass to the model.

cols = cols.drop('Specialization_Marketing Management',1)

logm5 = LogisticRegression()



#Pass the columns to generate the model and print summary

res = gen_model(logm5, cols)



# Check the VIF for the features

calcVIF(cols).head(3)
# Dropping the next unwanted variable to pass to the model.

cols = cols.drop('Lead Source_Reference',1)

logm6 = LogisticRegression()



#Pass the columns to generate the model and print summary

res = gen_model(logm6, cols)



# Check the VIF for the features

calcVIF(cols).head(3)
# Dropping the next unwanted variable to pass to the model.

cols = cols.drop('Page Views Per Visit',1)

logm7 = LogisticRegression()



#Pass the columns to generate the model and print summary

res = gen_model(logm7, cols)



# Check the VIF for the features

calcVIF(cols).head(3)
# # Dropping the next unwanted variable to pass to the model.

# cols = cols.drop('',1)

# logm8 = LogisticRegression()



# #Pass the columns to generate the model and print summary

# res = gen_model(logm8, cols)



# # Check the VIF for the features

# calcVIF(cols).head(3)
# Getting the predicted values on the train set



X_train_sm = sm.add_constant(X_train[cols])

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})

y_train_pred_final['Prospect ID'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['Predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
from sklearn.metrics import classification_report
print (classification_report(y_train_pred_final['Converted'], y_train_pred_final['Predicted']))
from sklearn import metrics

from sklearn.metrics import confusion_matrix
def get_metrics(actual, predicted):

    confusion = confusion_matrix(actual, predicted)



    # Let's check the overall accuracy.

    Accuracy = metrics.accuracy_score(actual, predicted)



    TN = confusion[0,0] # true negatives

    FP = confusion[0,1] # false positives

    FN = confusion[1,0] # false negatives

    TP = confusion[1,1] # true positive 



    # Calculate the different Metrics

    Sensitivity = TP / float(TP+FN) # calculate Sensitivity

    Specificity = TN / float(TN+FP) # calculate specificity

    Precision   = TP / float(TP+FP) # calculate Precision

    Recall      = TN / float(TN+FP) # calculate Recall

    FPR = (FP/ float(TN+FP))        # Calculate False Postive Rate - predicting conversion when customer does not convert

    PPV = (TP / float(TP+FP))       # positive predictive value 

    NPV = (TN / float(TN+ FN))      # Negative predictive value

    

    F1 = 2*(Precision*Recall)/(Precision+Recall)



    # Print the Metrics

    print (f'The Confusion Matrix is \n {confusion}')

    print (f'The Accuracy is    : {round (Accuracy,2)} ({Accuracy})')

    print (f'The Sensitivity is : {round (Sensitivity,2)} ({Sensitivity})')

    print (f'The Specificity is : {round (Specificity,2)} ({Specificity})')

    print (f'The Precision is   : {round (Precision, 2)} ({Precision})')

    print (f'The Recall is      : {round (Recall, 2)} ({Recall})')

    print (f'The f1 score is    : {round (F1, 2)} ({F1})')

    print (f'The False Positive Rate is       : {round (FPR, 2)} ({FPR})')

    print (f'The Positive Predictive Value is : {round (PPV, 2)} ({PPV})')

    print (f'The Negative Predictive Value is : {round (NPV, 2)} ({NPV})')

def plot_confusion_metrics(actual, predicted):

    sns.set_style('white')

    cm = confusion_matrix(actual, predicted)

    plt.clf()

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)

    classNames = ['Negative','Positive']

    plt.title('True Converted and Predicted Converted Confusion Matrix', fontsize=14)

    plt.ylabel('True Converted', fontsize=14)

    plt.xlabel('Predicted Converted', fontsize=14)

    tick_marks = np.arange(len(classNames))

    plt.xticks(tick_marks, classNames, fontsize=14)

    plt.yticks(tick_marks, classNames, fontsize=14)

    s = [['TN','FP'], ['FN', 'TP']]

    for i in range(2):

        for j in range(2):

            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]), fontsize=14, ha='center')

    plt.show()
get_metrics(y_train_pred_final.Converted, y_train_pred_final.Predicted)
plot_confusion_metrics(y_train_pred_final.Converted, y_train_pred_final.Predicted)
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic')

    plt.legend(loc="lower right")

    plt.show()
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, 

                                          y_train_pred_final.Converted_prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
numbers
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['probability','accuracy','sensitivity','specificity'])

from sklearn.metrics import confusion_matrix



#     TN = confusion[0,0] # true negatives

#     FP = confusion[0,1] # false positives

#     FN = confusion[1,0] # false negatives

#     TP = confusion[1,1] # true positive 

    

for i in numbers:

    cm1 = metrics.confusion_matrix(y_train_pred_final['Converted'], y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i , accuracy, sensitivity, specificity]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.



sns.set_style("whitegrid") # white/whitegrid/dark/ticks

sns.set_context("paper") # talk/poster

cutoff_df.plot.line(x='probability', y=['accuracy','sensitivity','specificity'], figsize=(10,6))



plt.xticks(np.arange(0, 1, step=0.05), size = 12)

plt.yticks(size = 12)

plt.title('Accuracy, Sensitivity and Specificity for various probabilities', fontsize=14)

plt.xlabel('Probability', fontsize=14)

plt.ylabel('Metrics', fontsize=14)

plt.show()
#### From the curve above, 0.36 is the optimum point to take it as a cutoff probability.



y_train_pred_final['final_Predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.36 else 0)

y_train_pred_final.head()
# Get all the necessary Metrics for the Training dataset for cut-off 0.36

print (f'The Final Evaluation Metrics for the train Dataset: ')

print (f'----------------------------------------------------')



get_metrics(y_train_pred_final['Converted'], y_train_pred_final['final_Predicted'])
# Plot Confusion metrics for final predicted for train data



plot_confusion_metrics(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)
# Classification report for the training dataset

print (classification_report(y_train_pred_final['Converted'], y_train_pred_final['final_Predicted']))
# Assign a Lead score based on the predictions



y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))



y_train_pred_final[['Converted','Converted_prob','Prospect ID','final_Predicted','Lead_Score']].head()
y_train_pred_final.head()
from sklearn.metrics import precision_recall_curve



p, r, thresholds = precision_recall_curve(y_train_pred_final['Converted'], y_train_pred_final['Converted_prob'])
# Plot the Precision / Recall tradeoff chart

sns.set_style("whitegrid") # white/whitegrid/dark/ticks

sns.set_context("paper") # talk/poster



plt.figure(figsize=(8, 4), dpi=100, facecolor='w', edgecolor='k', frameon='True')

plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.xticks(np.arange(0, 1, step=0.05))

plt.title('Precision and Recall for various probabilities', fontsize=14)

plt.xlabel('Probability', fontsize=14)

plt.ylabel('Metrics', fontsize=14)

plt.show()
X_test.head()
# Fit the Numeric features of the Test dataset with the Scaler method

X_test[numFeatures] = scaler.transform(X_test[numFeatures])

X_test.head()
X_test.shape
cols
# Making Predictions on the X_test dataset



X_test = X_test[cols]

X_test_sm = sm.add_constant(X_test)

X_test.head()
y_test_pred = res.predict(X_test_sm)
y_test_pred[:5]
# Converting y_pred to a dataframe from an array

y_test_pred_df = pd.DataFrame(y_test_pred)



# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)



# Putting CustID to index

y_test_pred_df['Prospect ID'] = y_test_df.index



# Removing index for both dataframes to append them side by side 

y_test_pred_df.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)



# Appending y_test_df and y_testest_pred_1

y_test_pred_final = pd.concat([y_test_df, y_test_pred_df],axis=1)



# Renaming the column 

y_test_pred_final= y_test_pred_final.rename(columns={ 0 : 'Converted_prob'})

y_test_pred_final.head(10)
# Rearranging the columns

y_test_pred_final = y_test_pred_final[['Prospect ID','Converted','Converted_prob']]

y_test_pred_final['Lead_Score'] = y_test_pred_final.Converted_prob.map( lambda x: round(x*100))

y_test_pred_final.head()
# Predict the final y values based on the threshold of 0.3

y_test_pred_final['final_Predicted'] = y_test_pred_final['Converted_prob'].map(lambda x: 1 if x > 0.36 else 0)



y_test_pred_final.head()
# Get all the necessary Metrics for the Test dataset 



print (f'The Final Evaluation Metrics for the test Dataset: ')

print (f'---------------------------------------------------')

get_metrics(y_test_pred_final['Converted'], y_test_pred_final['final_Predicted'])
# Plot Confusion metrics for final predicted for test data



plot_confusion_metrics(y_test_pred_final.Converted, y_test_pred_final.final_Predicted)
# Print the classification report for the Test Dataset

print (classification_report(y_test_pred_final['Converted'], y_test_pred_final['final_Predicted']))
y_train_pred_final.head()
# Create Dataset with y_train Prospect ID and Lead score

y_train_score = y_train_pred_final[['Prospect ID','Lead_Score']]



# Create Dataset with y_test Prospect ID and Lead score

y_test_score = y_test_pred_final[['Prospect ID','Lead_Score']]



# Concatenate the y_train scores and the y_test scores

df_score = pd.concat([y_train_score, y_test_score], ignore_index=True)



# Set the index of the final score dataset as the Prospect ID to concatenate the score dataset to the original data

df_score.set_index('Prospect ID', inplace=True)



# Inner Join the Original Leads dataset with the scores dataset. This will add a new column 'Lead_Score' to the 

# Original dataset. 

df_orig = df_orig.join(df_score['Lead_Score'])



df_orig.head()
pd.options.display.float_format = '{:.2f}'.format

model_params = res.params[1:]

model_params
#feature_importance = abs(new_params)



feature_importance = model_params

feature_importance = 100.0 * (feature_importance / feature_importance.max())

feature_importance
# Sort the feature variables based on their relative coefficient values



sorted_idx = np.argsort(feature_importance,kind='quicksort',order='list of str')

sorted_idx
# Plot to show the realtive Importance of each feature in the model 

pos = np.arange(sorted_idx.shape[0]) + .5



fig = plt.figure(figsize=(10,6))

ax = fig.add_subplot(1, 1, 1)

ax.barh(pos, feature_importance[sorted_idx], align='center', color = 'tab:blue',alpha=0.8)

ax.set_yticks(pos)

ax.set_yticklabels(np.array(X_train[cols].columns)[sorted_idx], fontsize=12)

ax.set_xlabel('Relative Feature Importance', fontsize=14)



plt.tight_layout()   

plt.show()