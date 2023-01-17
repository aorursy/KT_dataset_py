# Author: Caleb Woy



import numpy as np # linear algebra

import pandas as pd # data processing

from scipy.stats import kurtosis, skew # checking distributions

import scipy.stats as stat # plotting, mostly

import scipy.spatial.distance as sp # Computing distances in kNN

import matplotlib.pyplot as pl # plotting

import seaborn as sb # plotting

import sklearn as sk # Regression modelling

import os # Reading data

import sklearn.model_selection # train and test splitting

import matplotlib.pylab as plt # plotting hyperparamter cost curves

import time # timing custom knn model

from sklearn.model_selection import RandomizedSearchCV # tuning hyperparams for complex models

from sklearn.metrics.scorer import make_scorer # defining custom model evaluation function

from sklearn.ensemble import GradientBoostingClassifier as gb # modelling

from sklearn.neighbors import KNeighborsClassifier # modelling

from sklearn.model_selection import GridSearchCV # tuning hyperparams for simple models

from sklearn.ensemble import RandomForestClassifier as rf # modelling

from sklearn.svm import SVC as sv # modelling
# So we can see some interesting output without truncation

pd.options.display.max_rows = 1000



path_to_data = "/kaggle/input/"



# Loading the training and test data sets into pandas

train_original = pd.read_csv(path_to_data + "/adult.data", names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',

                                                                  'marital-status', 'occupation', 'relationship', 'race', 'sex',

                                                                  'cap-gain', 'cap-loss', 'hrsperwk', 'native', 'label'])

test_original = pd.read_csv(path_to_data + "/adult.test", skiprows=1, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',

                                                                             'marital-status', 'occupation', 'relationship', 'race', 'sex',

                                                                             'cap-gain', 'cap-loss', 'hrsperwk', 'native', 'label'])



# Combining the training and test sets

frames = [train_original, test_original]

data_original = pd.concat(frames)



# print the head

data_original.head()
feature_name = 'label'



# Checking counts per designation

data_original.groupby(feature_name).count()
# the output here is erroneously grouped into 4 rows, I need to remove the period from every label in the test set to get an accurate count.

data_original[[feature_name]] = data_original[[feature_name]].replace([" <=50K.", " >50K."], [" <=50K", " >50K"])

data_original.groupby(feature_name).count()
# better, now we can view a summary

data_original[[feature_name]].describe()
# So, there are 48842 values in the label column. There are 2 factor levels for the column. The most common label is '<= 50K' and it occurs 37155 times, 

# roughly 3/4 of the individuals.

# Now we'll check for missing values.

boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)

print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')

print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')
# No missing values for our label, that's good. I'll move onto the next feature.
feature_name = 'age'



# viewing a summary

data_original[[feature_name]].describe()
#checking for missing values

boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)

print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')

print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')
# No missing values on age, let's check skewness and kurtosis

print(f'Skewness: {skew(data_original.age)}')

print(f'Kurtosis: {kurtosis(data_original.age)}')
# The sample distribution of ages appears to be slightly right skewed with very slight negative kurtosis. This may need transformed for future modelling.

# Let's visualize this one to confirm the skewness.

x = data_original.age

pl.hist(x, bins=80)

pl.xlabel('Age')

pl.ylabel('Frequency')
# The values at the end of the right tail are definitely outliers however they're meaningful in our analysis (the elderly are important too). There don't appear to be any obvious

# errors caused by typos (like 500 or 0) 



# I'll create a new feature by taking the log 

# I'll create a new feature by centering with the z score

# I'll create a new feature by taking the log and centering with the z score



logage = np.log(data_original['age'])

data_original['log_age'] = logage



mean = np.mean(data_original['age'])

stdev = np.std(data_original['age'])

data_original['age_ZCentered'] = (data_original['age'] - mean) / stdev



mean = np.mean(logage)

stdev = np.std(logage)

data_original['log_age_ZCentered'] = (logage - mean) / stdev



x = data_original['log_age_ZCentered']

pl.hist(x, bins=80)

pl.xlabel('log_age_ZCentered')

pl.ylabel('Frequency')



# checking for success

data_original.head()



#all good
feature_name = 'workclass'



# viewing a summary

data_original[[feature_name]].describe()
# Roughly 3/4 of our individuals appear to be working in the private sector. Describe returned that there are 9 factor levels in this feature when we know there are actually 

# only 8. so there must be missing values in this feature. Let's check.



boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)

print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')

print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')

# There are 2799 ? values currently. None of them are Null or NaN values, so that's good. We have a few options here. The first is to impute the mode level (Private). 

# The second is to check if there are any other features here that might explain variation in with workclass, then if so, predict the missing workclass values. The third is 

# leave the ? value in as a placeholder unkown value and predict based on the effect of the level as we would any other feature.



# I'll make some boxplots to see if there's any explainable variation.



sb.boxplot( x=data_original["log_age_ZCentered"], y=data_original["workclass"] )

sb.boxplot( x=data_original["fnlwgt"], y=data_original["workclass"] )

sb.boxplot( x=data_original["education-num"], y=data_original["workclass"] )

sb.boxplot( x=data_original["cap-gain"], y=data_original["workclass"] )

sb.boxplot( x=data_original["cap-loss"], y=data_original["workclass"] )

sb.boxplot( x=data_original["hrsperwk"], y=data_original["workclass"] )
# None of these give off the appearance of explainatory variation that I'm looking to test with ANOVA so I'll impute the mode (Private) for the missing values. This can

# always be undone later during the modelling fase should we like to check how well we can predict with an unkown value effect.



# Checking the original counts at each factor level

data_original.groupby(feature_name).count()



# Making the replacement and recalculating the values

data_original[[feature_name]] = data_original[[feature_name]].replace([" ?"], [" Private"])

data_original.groupby(feature_name).count()



# All good.
feature_name = 'fnlwgt'



# viewing a summary

data_original[[feature_name]].describe()
# These are large numbers, any predictive model we apply on this data set would befit from some regularization here in the future. The max is exponetially larger than the mean.

# High values in fnlwgt will need investigated.



# Let's check for missing values.

boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)

print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')

print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')
# None, the data set authors created this feature so that should have been expected. Thanks authors!



# Checking skewness and kurtosis.

print(f'Skewness: {skew(data_original[feature_name])}')

print(f'Kurtosis: {kurtosis(data_original[feature_name])}')
# The fnlwgt column has some strong right skew and high positive kurtosis. It should look like a big spike on the left side of the distribution.



# Let's visualize to confirm.

x = data_original[feature_name]

pl.hist(x, bins=100)

pl.xlabel('fnlwgt')

pl.ylabel('Frequency')

# Yup. This feature would benefit from a log transformation. 



# Creating new features. I'll take the log transform then standardize that using the z-score.

logfnlwgt = np.log(data_original['fnlwgt'])

data_original['log_fnlwgt'] = logfnlwgt



mean = np.mean(data_original['fnlwgt'])

stdev = np.std(data_original['fnlwgt'])

data_original['fnl_wgt_ZCentered'] = (data_original['fnlwgt'] - mean) / stdev



mean = np.mean(logfnlwgt)

stdev = np.std(logfnlwgt)

data_original['log_fnl_wgt_ZCentered'] = (logfnlwgt - mean) / stdev



x = data_original['log_fnl_wgt_ZCentered']

pl.hist(x, bins=100)

pl.xlabel('log_fnl_wgt_ZCentered')

pl.ylabel('Frequency')



# checking for success

data_original.head()



# all good
#Let's view the largest values of the distribution. 

data_original.nlargest(10, ['log_fnl_wgt_ZCentered']) 
# Regarding the outliers at the tail of fnlwgt, none of these appear to be abnormal. We can't know forsure 

# without knowing how fnlwgt was calulated, yet the consistent increasing of the feature values up to the max appears systematic and not erroneous. I won't do anything

# special about them.
feature_name = 'education'



# viewing a summary

data_original[[feature_name]].describe()
# Seeing the correct number of unique factor levels here so likely no missing values. HS-grad is the most common level

# Let's confirm:



boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)

print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')

print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')
# Yup. Looking good here.
feature_name = 'education-num'



# viewing a summary

data_original[[feature_name]].describe()
# Mean value of ~10. Max and min are prestty evenly spread.

# Checking for missing values:



boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)

print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')

print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')
# Let's look at the skewness and kurtosis:



print(f'Skewness: {skew(data_original[feature_name])}')

print(f'Kurtosis: {kurtosis(data_original[feature_name])}')
# Slight positive kurtosis, slight left skew. Let's visualize:



x = data_original[feature_name]

pl.hist(x, bins=16)

pl.xlabel('education-num')

pl.ylabel('Frequency')
# This distribution appears bimodal. Likely due to the effect of college. This might make the categorical feature (education) more useful to us.



# I'll scale this by transforming it with the Z-score.



mean = np.mean(data_original[feature_name])

stdev = np.std(data_original[feature_name])

education_num_ZCentered = (data_original[feature_name] - mean) / stdev



# Visualizing:

x = education_num_ZCentered

pl.hist(x, bins=16)

pl.xlabel('education_num_ZCentered')

pl.ylabel('Frequency')
# Now to replace the original feature with the transformed version.



data_original['education_num_ZCentered'] = education_num_ZCentered



# Checking:

data_original.head()
feature_name = 'marital-status'



# viewing a summary

data_original[[feature_name]].describe()
# There are 7 unique factor levels present in our distribution. So, likely no missing values. We can confirm.



boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)

print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')

print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')
feature_name = 'occupation'



# viewing a summary

data_original[[feature_name]].describe()
# There are only supposed to be 14 factor levels so they're definitely some missing values here.



boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)

print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')

print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')
# 2809 missing values. I'll impute the mode value (Prof-specialty)



data_original.groupby(feature_name).count()



# Making the replacement and recalculating the values

data_original[[feature_name]] = data_original[[feature_name]].replace([" ?"], [" Prof-specialty"])

data_original.groupby(feature_name).count()



# Worked fine.
feature_name = 'relationship'



# viewing a summary

data_original[[feature_name]].describe()
# Seeing 6 unique factor levels as expected. Most common level is Husband.

# Confirming no missing values



boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)

print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')

print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')
# Good.
feature_name = 'race'



# viewing a summary

data_original[[feature_name]].describe()
# Seeing 5 unique factor levels as expected. Most common level is White.

# Confirming no missing values



boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)

print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')

print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')
# Good.
feature_name = 'sex'



# viewing a summary

data_original[[feature_name]].describe()
# Seeing 2 unique factor levels as expected. Most common level is Male.

# Confirming no missing values



boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)

print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')

print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')
# Good.
feature_name = 'cap-gain'



# viewing a summary

data_original[[feature_name]].describe()
# The summary here tells us the mean gain is about a thousand dollars. The distribution appears to be dramatically right skewed and is likely mostly (0) values.

# Let's comfirm by checking skew and kurtosis.



print(f'Skewness: {skew(data_original[feature_name])}')

print(f'Kurtosis: {kurtosis(data_original[feature_name])}')
# Yeah . . . we'll be transforming this one. First, Let's check for missing values.



boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)

print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')

print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')
# None. That's helpful. Time to visualize this one.



x = data_original[feature_name]

pl.hist(x, bins=100)

pl.xlabel('cap_gain')

pl.ylabel('Frequency')
# Before I transform this, I want to investigate the outlier here that's near 100K.



data_original.nlargest(10, [feature_name]) 



# I've actually viewed the top 60 largest values here but I set the code back to outputting the top 10 to keep this report cleaner.

# The trend shown in the top 10 is the same as the top 60, all are labeled > 50k. 
# I think I want to create a new feature here. A simple binary feature specifying whether the individual made > 50K in capital gains alone.



data_original['cap-gains50k'] = data_original.apply(lambda x: True if x[feature_name] > 50000 else False, axis=1).astype('category')



# Checking that it worked:

data_original.nlargest(10, [feature_name]) 
# That should be a really significant factor in whatever model we might use to predict our label.

# Now I'll transform the original cap-gain feature. Taking a log of 0 will produce NaNs so I'll transform the feature to log(cap-gains + 1) and then I'll scale It with the z-score.



log_cap_gain = np.log(data_original[feature_name] + 1)

data_original['log_cap_gain'] = log_cap_gain



# Visualizing:

x = log_cap_gain

pl.hist(x, bins=100)

pl.xlabel('log_cap_gain')

pl.ylabel('Frequency')
# Now scaling by Z-score:



mean = np.mean(data_original[feature_name])

stdev = np.std(data_original[feature_name])

data_original['cap_gain_ZCentered'] = (data_original[feature_name] - mean) / stdev



mean = np.mean(log_cap_gain)

stdev = np.std(log_cap_gain)

data_original['log_cap_gain_ZCentered'] = (log_cap_gain - mean) / stdev



# Visualizing:

x = data_original['log_cap_gain_ZCentered']

pl.hist(x, bins=100)

pl.xlabel('log_cap_gain_ZCentered')

pl.ylabel('Frequency')
# Checking:

data_original.head()
# Good.
feature_name = 'cap-loss'



# viewing a summary

data_original[[feature_name]].describe()
# This distribution appears that it'll be similar to cap-gain. However, the max loss is far less than 50K so I don't think I'll be making a new feature representing this one.

# Let's check for missing values:



boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)

print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')

print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')
# Good. Now skewness and kurtosis:



print(f'Skewness: {skew(data_original[feature_name])}')

print(f'Kurtosis: {kurtosis(data_original[feature_name])}')
# Yeah pretty bad. Time to visualize:



x = data_original[feature_name]

pl.hist(x, bins=100)

pl.xlabel('log_cap_gain')

pl.ylabel('Frequency')
# I know the max value, it's definitly an outlier. Let's invetigate for others.



data_original.nlargest(10, [feature_name]) 
# These all look proper. I'll apply the same transformation to this as I did on cap-gain to keep it consistently scaled with the rest of our features.



log_cap_loss = np.log(data_original[feature_name] + 1)

data_original['log_cap_loss'] = log_cap_loss



mean = np.mean(data_original[feature_name])

stdev = np.std(data_original[feature_name])

cap_loss_ZCentered = (data_original[feature_name] - mean) / stdev



mean = np.mean(log_cap_loss)

stdev = np.std(log_cap_loss)

log_cap_loss_ZCentered = (log_cap_loss - mean) / stdev



# Visualizing:

x = log_cap_loss_ZCentered

pl.hist(x, bins=100)

pl.xlabel('log_cap_loss_ZCentered')

pl.ylabel('Frequency')
# Now to replace the original feature with the transformed version.



data_original['cap_loss_ZCentered'] = cap_loss_ZCentered

data_original['log_cap_loss_ZCentered'] = log_cap_loss_ZCentered



# Checking:

data_original.head()
# Good.
feature_name = 'hrsperwk'



# viewing a summary

data_original[[feature_name]].describe()
# Mean is about 40 hours, as expected. The first and third quartile are pretty tight to the mean so we'll likely see high kurtosis here. Probably some minor right skew too.

# I'll check skewness and kurtosis, as well as for missing values:



print(f'Skewness: {skew(data_original[feature_name])}')

print(f'Kurtosis: {kurtosis(data_original[feature_name])}')



boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)

print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')

print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')
# Yup. Glad there aren't missing values.

# Let's visualize:



x = data_original[feature_name]

pl.hist(x, bins=100)

pl.xlabel('hrsperwk')

pl.ylabel('Frequency')
# I'll just Z-center this one to scale it properly. The skewness here isn't that extreme.



mean = np.mean(data_original[feature_name])

stdev = np.std(data_original[feature_name])

hrs_per_wk_ZCentered = (data_original[feature_name] - mean) / stdev



# Now to add the transformed version.



data_original['hrs_per_wk_ZCentered'] = hrs_per_wk_ZCentered



# Checking:

data_original.head()
feature_name = 'native'



# viewing a summary

data_original[[feature_name]].describe()
# Our summary tells us there are 42 unique factor levels here. However, there are only 41 listed in the description, so we have missing values. Most common value is United-states

# Confirming:



boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)

print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')

print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')
# I'll impute the mode (United-States) for the missing values.



data_original.groupby(feature_name).count()



# Making the replacement and recalculating the values

data_original[[feature_name]] = data_original[[feature_name]].replace([" ?"], [" United-States"])

data_original.groupby(feature_name).count()



# Worked fine.
# Checking correlation between all numeric features.



correlation_matrix = data_original.corr().round(2)

pl.figure(figsize=(10,8))

sb.heatmap(data=correlation_matrix, annot=True, center=0.0, cmap='coolwarm')
# Our numeric features have very weak correlation to eachother, despite the strong correlations between features and their transformed features that I created. 

# This is good for us if we're looking to predict our label because it means we won't have to worry about the preoblem of multicollinearity among them. 
# Each graph will be created with the same code, I'll just switch the names of the variables hue_lab and x_lab

hue_lab = 'workclass'

x_lab = 'education'



# Grouping by the hue_group, then counting by the x_lab group

hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)



# Creating the percentage vector to measure the frequency of each type

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]



# Creating and plotting the new dataframe 

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'workclass'

x_lab = 'occupation'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'cap-gains50k'

x_lab = 'workclass'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'cap-gains50k'

x_lab = 'education'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'marital-status'

x_lab = 'occupation'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'relationship'

x_lab = 'marital-status'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'sex'

x_lab = 'marital-status'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'cap-gains50k'

x_lab = 'marital-status'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'relationship'

x_lab = 'occupation'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'sex'

x_lab = 'occupation'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'cap-gains50k'

x_lab = 'occupation'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'race'

x_lab = 'relationship'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'sex'

x_lab = 'relationship'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'cap-gains50k'

x_lab = 'relationship'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'cap-gains50k'

x_lab = 'sex'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
sb.boxplot( x=data_original["log_age_ZCentered"], y=data_original["workclass"] )
sb.boxplot( x=data_original["log_age_ZCentered"], y=data_original["education"] )
sb.boxplot( x=data_original["log_age_ZCentered"], y=data_original["marital-status"] )
sb.boxplot( x=data_original["log_age_ZCentered"], y=data_original["cap-gains50k"] )
sb.boxplot( x=data_original["log_fnl_wgt_ZCentered"], y=data_original["race"] )
sb.boxplot( x=data_original["education_num_ZCentered"], y=data_original["workclass"] )
sb.boxplot( x=data_original["education_num_ZCentered"], y=data_original["occupation"] )
sb.boxplot( x=data_original["education_num_ZCentered"], y=data_original["race"] )
sb.boxplot( x=data_original["education_num_ZCentered"], y=data_original["cap-gains50k"] )
sb.boxplot( x=data_original["hrs_per_wk_ZCentered"], y=data_original["workclass"] )
sb.boxplot( x=data_original["hrs_per_wk_ZCentered"], y=data_original["education"] )
sb.boxplot( x=data_original["hrs_per_wk_ZCentered"], y=data_original["marital-status"] )
sb.boxplot( x=data_original["hrs_per_wk_ZCentered"], y=data_original["occupation"] )
sb.boxplot( x=data_original["hrs_per_wk_ZCentered"], y=data_original["relationship"] )
sb.boxplot( x=data_original["hrs_per_wk_ZCentered"], y=data_original["sex"] )
sb.boxplot( x=data_original["hrs_per_wk_ZCentered"], y=data_original["cap-gains50k"] )
sb.boxplot( x=data_original["log_age_ZCentered"], y=data_original["label"] )
sb.boxplot( x=data_original["education_num_ZCentered"], y=data_original["label"] )
sb.boxplot( x=data_original["log_fnl_wgt_ZCentered"], y=data_original["label"] )
sb.boxplot( x=data_original["log_cap_gain_ZCentered"], y=data_original["label"] )
sb.boxplot( x=data_original["log_cap_loss_ZCentered"], y=data_original["label"] )
sb.boxplot( x=data_original["hrs_per_wk_ZCentered"], y=data_original["label"] )
hue_lab = 'label'

x_lab = 'workclass'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'label'

x_lab = 'education'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'label'

x_lab = 'marital-status'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'label'

x_lab = 'occupation'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'label'

x_lab = 'relationship'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'label'

x_lab = 'race'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'label'

x_lab = 'sex'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
hue_lab = 'cap-gains50k'

x_lab = 'label'



hue_group = data_original.groupby([hue_lab], sort=False)

counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

data = [

    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 

    (hue, x), percentage in dict(counts).items()

]

df = pd.DataFrame(data)

p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);

p.set_xticklabels(rotation=90)

p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')
# Finally, dropping any duplicate rows

first_len = len(data_original)

data_original.drop_duplicates()

print(f'Dropped {first_len - len(data_original)} records.')



# Final data set

data_original.head()
# splitting off the test set

print(len(train_original), end = '\t')

print(len(test_original))



holdout = data_original.tail(16281)

data_original = data_original.head(32561)
"""

Definition of custom scoring function that utilizes our scoring metric.



y: array, actual test labels

y_pred: array, predicted labels

"""

def score(y, y_pred):

    score = 0

    for x1, x2 in zip(y, y_pred):

        # increase score by 1 for every true positive

        if x2 == 1 and x1 == x2:

            score += 1

        elif x2 == 1 and x1 != x2:

            # decrease score by 1 for every false positive

            score -= 1

    return score
"""

Utilizing GridSearchCV's parallel processing to speed up the process of finding the optimal k value.

Maximum number of available processors will be used.



data: pd dataframe, the features and labels

cat_columns: array, names of categorical columns to create dummy encodings for

distance: str, metric to be used in distance calculation



"""

def fit_sklearn_knn_hyperparams(data, cat_columns, score_function, distance = 'euclidean'):



    # initialize scorer function, compatible with GridSearchCV

    my_scorer = make_scorer(score_function, greater_is_better=True)



    data_knn_full = data

    # create dummy encodings of categorical features

    data_knn_full = pd.get_dummies(data_knn_full, columns=cat_columns)



    # define the values of k to test

    k = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]

    grid_param = {'n_neighbors': k}



    # initialize model with given distance metric

    model = KNeighborsClassifier(metric = distance)

    # initialize grid search with custom scoring function, using default number of folds (5)

    KNN_random = GridSearchCV(estimator = model, 

                                 param_grid = grid_param,

                                 scoring = my_scorer,

                                 verbose = 2,

                                 cv = 5,

                                 n_jobs = -1)

    # begin tuning

    KNN_random.fit(data_knn_full.drop(columns=['label']), data_knn_full['label'].replace([" <=50K", " >50K"], [0, 1]))

    # print results

    print(KNN_random.best_params_)

    print(f'score: {KNN_random.best_score_}')
"""

Compute and print the cost matrix for a single k value. Utilizes sklearn kNN.



train: pd dataframe, training data

test: pd dataframe, testing data

cat_columns: array, names of categorical columns to create dummy encodings for

distance: str, metric to be used in distance calculation

k: int, the number of neighbors to tally a vote with



"""

def test_sklearn_knn(train, test, cat_columns, distance='euclidean', k = 1):

    # define number of expiriments to perform and initialize confusion matrix

    result_avgs = [0, 0, 0, 0]



    # create dummy encodings of categorical features

    train = pd.get_dummies(train, columns=cat_columns)

    test = pd.get_dummies(test, columns=cat_columns)



    # seperate features from labels, replace labels with 0s and 1s

    xtrain = train.drop(columns=['label'])

    xtest = test.drop(columns=['label'])

    ytrain = train['label'].replace([" <=50K", " >50K"], [0, 1])

    ytest = test['label'].replace([" <=50K", " >50K"], [0, 1])



    # predict result matrix

    knn = KNeighborsClassifier(n_neighbors = k, metric = distance)

    knn.fit(xtrain, ytrain)

    result = knn.predict(xtest)



    # create simple list of test labels

    y = ytest.values.tolist()



    # iterate over the results for fixed k value and increment counts for each metric of the confusion matrix

    count_true_pos, count_false_pos, count_true_neg, count_false_neg = 0, 0, 0, 0

    for j in range(len(result)):

        if y[j] == 1:

            if result[j] == y[j]:

                count_true_pos += 1

            else:

                count_false_neg += 1

        else:

            if result[j] == y[j]:

                count_true_neg += 1

            else:

                count_false_pos += 1

                

    # bin the counts

    result_avgs[0] += count_true_pos

    result_avgs[1] += count_false_pos

    result_avgs[2] += count_true_neg

    result_avgs[3] += count_false_neg



    # print confusion matrix

    print()     

    print(f'CONFUSION MATRIX FOR K = {k}: ')

    print(f'*********************************')

    print(f'| TP = {round(result_avgs[0], 2)} \t| FP = {round(result_avgs[1], 2)} \t|')

    print(f'*********************************')

    print(f'| FN = {round(result_avgs[3], 2)} \t| TN = {round(result_avgs[2], 2)} \t|')

    print(f'*********************************')

    print(f'SCORE ON SAMPLE: {result_avgs[0] - result_avgs[1]}')

    print(f'ACCURACY: {(result_avgs[0] + result_avgs[2])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}', end='\t')

    print(f'ALPHA: {(result_avgs[1])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}', end='\t')

    print(f'BETA: {(result_avgs[3])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}')

    print()
#Fitting model with raw data

fit_sklearn_knn_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age', 'hrsperwk']], 

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], score)
#Fitting model with log transformed data

fit_sklearn_knn_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age', 'hrsperwk']], 

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], score)
#Fitting model with z-score centered data

fit_sklearn_knn_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']], 

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], score)
#Fitting model with log and z-score centered data

fit_sklearn_knn_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']], 

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], score)
# I'll test both the z-score only model and the log + z-score model because they're so close in cross validation.
#Testing model with z-score centered data, k = 40

test_sklearn_knn(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']], 

                 holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']],

                 ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], distance = 'euclidean', k = 40)
#Testing model with log and z-score centered data, k = 50

test_sklearn_knn(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']], 

                 holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']],

                 ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], distance = 'euclidean', k = 50)
#Fitting model with just categorical features

fit_sklearn_knn_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass']], 

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], score, 'jaccard')
#Testing model with just categorical features, k = 70, distance = 'jaccard'

test_sklearn_knn(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass']], 

                 holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass']],

                 ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 'jaccard', k = 70)
"""

Utilizing RandomizedSearchCV's parallel processing to speed up the process of finding the optimal values of n_estimators, min_samples_split, and min_samples_leaf.

Fitting a Random Forest.

Maximum number of available processors will be used.



data: pd dataframe, the features and labels

cat_columns: array, names of categorical columns to create dummy encodings for

random_seed_adder: int, value to be used in calculating random seed of train-test split



"""

def fit_sklearn_rf_hyperparams(data, cat_columns, random_seed_adder, score_function):

    # initialize scorer function, compatible with RandomizedSearchCV

    my_scorer = make_scorer(score_function, greater_is_better=True)



    data_rf_full = data

    # create dummy encodings of categorical features

    data_rf_full = pd.get_dummies(data_rf_full, columns=cat_columns)



    # define the values of n_estimators, min_samples_split, min_samples_leaf to test

    # n_estimators effects the bias of the model

    # min_samples_split and min_samples_leaf mainly effect model variance

    n_estimators = [100, 200, 500]

    min_samples_split = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

    min_samples_leaf = [1, 2, 5, 10, 15, 20, 25, 30]

    grid_param = {'n_estimators': n_estimators,

                  'min_samples_split': min_samples_split,

                  'min_samples_leaf': min_samples_leaf}

    

    # initialize model

    model = rf(random_state=1)

    # initialize randomized search with custom scoring function, using default number of folds (5)

    RFC_random = RandomizedSearchCV(estimator = model, 

                                 param_distributions = grid_param,

                                 n_iter = 100,

                                 scoring = my_scorer,

                                 verbose=2,

                                 cv = 5,

                                 random_state = random_seed_adder,

                                 n_jobs = -1)

    # begin tuning

    RFC_random.fit(data_rf_full.drop(columns=['label']), data_rf_full['label'].replace([" <=50K", " >50K"], [0, 1]))

    # print results

    print(RFC_random.best_params_)

    print(f'score: {RFC_random.best_score_}')
"""

Compute and print the cost matrix for a single hyperparameter setting. Utilizes sklearn Random Forest.



train: pd dataframe, training data

test: pd dataframe, testing data

cat_columns: array, names of categorical columns to create dummy encodings for

n_trees: int, number of trees to generate

m_min: int, min number of samples required to split a node

m_leave: int, min number of samples required to be at each leaf 



"""

def test_sklearn_rf(train, test, cat_columns, n_trees, m_min, m_leave):

    # define number of expiriments to perform and initialize confusion matrix

    result_avgs = [0, 0, 0, 0]



    # create dummy encodings of categorical features

    train = pd.get_dummies(train, columns=cat_columns)

    test = pd.get_dummies(test, columns=cat_columns)



    # seperate features from labels, replace labels with 0s and 1s

    xtrain = train.drop(columns=['label'])

    xtest = test.drop(columns=['label'])

    ytrain = train['label'].replace([" <=50K", " >50K"], [0, 1])

    ytest = test['label'].replace([" <=50K", " >50K"], [0, 1])



    # initialize model and predict result matrix

    rf_model = rf(n_estimators = n_trees, min_samples_split = m_min, min_samples_leaf = m_leave)

    rf_model.fit(xtrain, ytrain)

    result = rf_model.predict(xtest)



    # create simple list of test labels

    y = ytest.values.tolist()



    # iterate over the results for fixed k value and increment counts for each metric of the confusion matrix

    count_true_pos, count_false_pos, count_true_neg, count_false_neg = 0, 0, 0, 0

    for j in range(len(result)):

        if y[j] == 1:

            if result[j] == y[j]:

                count_true_pos += 1

            else:

                count_false_neg += 1

        else:

            if result[j] == y[j]:

                count_true_neg += 1

            else:

                count_false_pos += 1

                

    # bin the counts

    result_avgs[0] += count_true_pos

    result_avgs[1] += count_false_pos

    result_avgs[2] += count_true_neg

    result_avgs[3] += count_false_neg



    # print confusion matrix

    print()     

    print(f'CONFUSION MATRIX FOR n_estimators = {n_trees}, min_samples_split = {m_min}, min_samples_leaf = {m_leave}: ')

    print(f'*********************************')

    print(f'| TP = {round(result_avgs[0], 2)} \t| FP = {round(result_avgs[1], 2)} \t|')

    print(f'*********************************')

    print(f'| FN = {round(result_avgs[3], 2)} \t| TN = {round(result_avgs[2], 2)} \t|')

    print(f'*********************************')

    print(f'SCORE ON SAMPLE: {result_avgs[0] - result_avgs[1]}')

    print(f'ACCURACY: {(result_avgs[0] + result_avgs[2])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}', end='\t')

    print(f'ALPHA: {(result_avgs[1])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}', end='\t')

    print(f'BETA: {(result_avgs[3])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}')

    print()
#Fitting model with raw data

fit_sklearn_rf_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age', 'hrsperwk']], 

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 654, score)
#Fitting model with log transformed data

fit_sklearn_rf_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age', 'hrsperwk']], 

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 96846, score)
#Fitting model with z-score centered data

fit_sklearn_rf_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']], 

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 496874, score)
#Fitting model with log and z-score centered data

fit_sklearn_rf_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']], 

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 564, score)
# Testing all because they're very close
#Testing model with raw data, n_estimators = 200, min_samples_split = 35, min_samples_leaf = 2

test_sklearn_rf(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age', 'hrsperwk']], 

                holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age', 'hrsperwk']],

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 200, 35, 2)
#Testing model with log transformed data, n_estimators = 200, min_samples_split = 35, min_samples_leaf = 2

test_sklearn_rf(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age', 'hrsperwk']], 

                holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age', 'hrsperwk']],

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 200, 35, 2)
#Testing model with z-score centered data, n_estimators = 500, min_samples_split = 35, min_samples_leaf = 2

test_sklearn_rf(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']], 

                holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']],

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 500, 35, 2)
#Testing model with log and z-score centered data, n_estimators = 500, min_samples_split = 35, min_samples_leaf = 2

test_sklearn_rf(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']], 

                holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']],

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 500, 35, 2)
"""

Utilizing RandomizedSearchCV's parallel processing to speed up the process of finding the optimal values of n_estimators, min_samples_split, and min_samples_leaf.

Fitting a Gradient Boosting Classifier.

Maximum number of available processors will be used.



data: pd dataframe, the features and labels

cat_columns: array, names of categorical columns to create dummy encodings for

random_seed_adder: int, value to be used in calculating random seed of train-test split



"""

def fit_sklearn_GB_hyperparams(data, cat_columns, random_seed_adder, score_function):

    # initialize scorer function, compatible with RandomizedSearchCV

    my_scorer = make_scorer(score_function, greater_is_better=True)



    # define the values of n_estimators, learning_rate, min_samples_split, min_samples_leaf to test

    # n_estimators effects the bias of the model

    # min_samples_split and min_samples_leaf mainly effect model variance

    n_estimators = [100, 250, 500, 750, 1000, 1250]

    learning_rate = [0.01, 0.05, 0.1, 0.2, 0.3]

    min_samples_split = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

    min_samples_leaf = [1, 2, 5, 10, 15, 20, 25, 30]

    grid_param = {'n_estimators': n_estimators,

                  'learning_rate': learning_rate,

                  'min_samples_split': min_samples_split,

                  'min_samples_leaf': min_samples_leaf}

    

    data_gb_full = data

    # create dummy encodings of categorical features

    data_gb_full = pd.get_dummies(data_gb_full, columns=cat_columns)



    # initialize model

    model = gb(random_state=1)

    # initialize randomized search with custom scoring function, using default number of folds (5)

    GB_random = RandomizedSearchCV(estimator = model, 

                                 param_distributions = grid_param,

                                 scoring = my_scorer,

                                 verbose=5,

                                 cv = 5,

                                 random_state = random_seed_adder,

                                 n_jobs = -1)

    # begin tuning

    GB_random.fit(data_gb_full.drop(columns=['label']), data_gb_full['label'].replace([" <=50K", " >50K"], [0, 1]))

    # print result

    print(GB_random.best_params_)

    print(f'score: {GB_random.best_score_}')
"""

Compute and print the cost matrix for a single hyperparameter setting. Utilizes sklearn Gradient Boosting Classifier.



train: pd dataframe, training data

test: pd dataframe, testing data

cat_columns: array, names of categorical columns to create dummy encodings for

n_trees: int, number of trees to generate

lr: float, the learning rate

m_min: int, min number of samples required to split a node

m_leave: int, min number of samples required to be at each leaf 



"""

def test_sklearn_gb(train, test, cat_columns, n_trees, lr, m_min, m_leave):

    # define number of expiriments to perform and initialize confusion matrix

    result_avgs = [0, 0, 0, 0]



    # create dummy encodings of categorical features

    train = pd.get_dummies(train, columns=cat_columns)

    test = pd.get_dummies(test, columns=cat_columns)



    # seperate features from labels, replace labels with 0s and 1s

    xtrain = train.drop(columns=['label'])

    xtest = test.drop(columns=['label'])

    ytrain = train['label'].replace([" <=50K", " >50K"], [0, 1])

    ytest = test['label'].replace([" <=50K", " >50K"], [0, 1])



    # initialize model and predict result matrix

    gb_model = gb(n_estimators = n_trees, learning_rate = lr, min_samples_split = m_min, min_samples_leaf = m_leave)

    gb_model.fit(xtrain, ytrain)

    result = gb_model.predict(xtest)



    # create simple list of test labels

    y = ytest.values.tolist()



    # iterate over the results for fixed k value and increment counts for each metric of the confusion matrix

    count_true_pos, count_false_pos, count_true_neg, count_false_neg = 0, 0, 0, 0

    for j in range(len(result)):

        if y[j] == 1:

            if result[j] == y[j]:

                count_true_pos += 1

            else:

                count_false_neg += 1

        else:

            if result[j] == y[j]:

                count_true_neg += 1

            else:

                count_false_pos += 1

                

    # bin the counts

    result_avgs[0] += count_true_pos

    result_avgs[1] += count_false_pos

    result_avgs[2] += count_true_neg

    result_avgs[3] += count_false_neg



    # print confusion matrix

    print()     

    print(f'CONFUSION MATRIX FOR n_estimators = {n_trees}, learning_rate = {lr}, min_samples_split = {m_min}, min_samples_leaf = {m_leave}: ')

    print(f'*********************************')

    print(f'| TP = {round(result_avgs[0], 2)} \t| FP = {round(result_avgs[1], 2)} \t|')

    print(f'*********************************')

    print(f'| FN = {round(result_avgs[3], 2)} \t| TN = {round(result_avgs[2], 2)} \t|')

    print(f'*********************************')

    print(f'SCORE ON SAMPLE: {result_avgs[0] - result_avgs[1]}')

    print(f'ACCURACY: {(result_avgs[0] + result_avgs[2])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}', end='\t')

    print(f'ALPHA: {(result_avgs[1])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}', end='\t')

    print(f'BETA: {(result_avgs[3])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}')

    print()
#Fitting model with raw data

fit_sklearn_GB_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age', 'hrsperwk']], 

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 5416, score)
#Fitting model with log transformed data

fit_sklearn_GB_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age', 'hrsperwk']], 

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 6748, score)
#Fitting model with z-score centered data

fit_sklearn_GB_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']], 

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 86784, score)
#Fitting model with log and z-score centered data

fit_sklearn_GB_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']], 

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 6874384, score)
# Testing all because they're very close
#Testing model with raw data, n_estimators = 750, learning_rate = 0.05, min_samples_split = 50, min_samples_leaf = 20

test_sklearn_gb(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age', 'hrsperwk']],

                holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age', 'hrsperwk']],

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 750, 0.05, 50, 20)
#Testing model with log transformed data, n_estimators = 500, learning_rate = 0.1, min_samples_split = 2, min_samples_leaf = 25

test_sklearn_gb(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age', 'hrsperwk']],

                holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age', 'hrsperwk']],

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 500, 0.1, 2, 25)
#Testing model with z-score centered data, n_estimators = 1250, learning_rate = 0.05, min_samples_split = 40, min_samples_leaf = 3

test_sklearn_gb(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']],

                holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']],

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 1250, 0.05, 40, 2)
#Testing model with log and z-score centered data, n_estimators = 500, learning_rate = 0.1, min_samples_split = 55, min_samples_leaf = 20

test_sklearn_gb(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']],

                holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']],

                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 750, 0.05, 20, 30)