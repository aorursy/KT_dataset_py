import pandas as pd, numpy as np

#Import flattened CSV census data into dataframe
df = pd.read_csv('../input/flattened_census_data.csv')

#Function to convert target from numeric to boolean
def cat_target(row):
    if row['over_50k'] == 0: #Less than $50k
        return False
    elif row['over_50k'] == 1: #Greater than $50k
        return True 
    else:
        return np.NaN
    
#Create categorical version of target variable
df['over_50_bool'] = df.apply(cat_target, axis=1)

#Print prevalance of respondents earning more than $50k a year
df['over_50_bool'].value_counts(normalize=True)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

#Store the independent variables we're interested in a list for looping
#through chart generation
cat_vars = ['race','sex','occupation','marital_status']

#Display by category the likelihood of a respondent earning over 50k
i=1
f, ax = plt.subplots(5,figsize=(16,16))

#Plot overall frequencies for Income Category
sns.barplot(x='over_50_bool', y="respondent_id", data=df, estimator=lambda x: len(x) / len(df) * 100,ax=ax[0])
ax[0].set(ylabel="Percent Frequency (%)")

#Plot % frequencies for categorical variables
for cat_var in cat_vars:
    sns.barplot(x=cat_var, y="respondent_id", data=df, estimator=lambda x: len(x) / len(df) * 100,ax=ax[i],hue="over_50_bool")
    ax[i].set(ylabel="Percent Frequency (%)")
    i+=1
ax[0].set_title("% Frequencies by Income Category for Categorical Variables")

#Store the independent variables we're interested in a list for looping
#through chart generation
numeric_vars = ['age','capital_gain','hours_week','education_num']

#Plot distributions for the numeric vars based on the Income Category
for numeric_var in numeric_vars:
    f, ax = plt.subplots(figsize=(16,4),ncols=2)
    over50=df[df.over_50_bool == True]
    under50=df[df.over_50_bool == False]
    ax[0].title.set_text('Over 50K: ' + numeric_var+' histogram')
    sns.distplot(over50[numeric_var],norm_hist=True,ax=ax[0])
    ax[1].title.set_text('Under 50K: ' + numeric_var+' histogram')
    sns.distplot(under50[numeric_var],norm_hist=True,ax=ax[1])
#Function to convert variable into a boolean (string)
def bool_my_feature(row,in_var,threshold):
    if row[in_var] > threshold:
        return 'True'
    elif row[in_var] <= threshold:
        return 'False' 
    else:
        return np.NaN

#Function to categorize numeric variable into bins
def bin_my_feature(row,in_var,bin_list):
    bin_list.sort() #Just in case
    out_label = np.NaN #If field doesn't match any bins
    n = len(bin_list)-1 #Max index
    for i in list(range(0,n+1)):
        if i < n:
            #Lower and upper ends of bin
            x,y = bin_list[i:i+2] 
        if i == 0 and row[in_var] <= bin_list[i]:
            #Value is less than min bin value
            out_label = '<=' + str(bin_list[i])
        elif i == n and row[in_var] >= bin_list[i]:
            #Value is greater than max bin value
            out_label = '>=' + str(bin_list[i])
        elif i < n and row[in_var] >= x and row[in_var] < y:
            out_label = str(x) + ' - ' + str(y)
    return out_label

#Create categorical versions of numeric independent variables
df['cap_gain_gt_0'] = df.apply(bool_my_feature, args=('capital_gain',0),axis=1)
df['education_num_gt_12'] = df.apply(bool_my_feature, args=('education_num',12),axis=1)
df['hours_week_gt_40'] = df.apply(bool_my_feature, args=('hours_week',40),axis=1)
df['age_bins'] = df.apply(bin_my_feature, args=('age',[20,30,40,50,60]),axis=1)

#Prepare variables for displaying newly created features
new_cat_features = ['cap_gain_gt_0','education_num_gt_12','hours_week_gt_40','age_bins']
i=0
f, ax = plt.subplots(4,figsize=(16,16))

#Generate plots for new features to assess information gain
for cat_var in new_cat_features:
    sns.barplot(x=cat_var, y="respondent_id", data=df, estimator=lambda x: len(x) / len(df) * 100,ax=ax[i],hue="over_50_bool")
    ax[i].set(ylabel="Percent Frequency (%)")
    i+=1
ax[0].set_title("% Frequencies by Income Category for Categorical Variables")
#Weights for Training and Test Datasets
train_weight = 0.8
test_weight = 0.2

#Sample records for test dataset 
test_bool = np.random.rand(len(df)) < test_weight
test = df[test_bool]

#Of remaining records not in test, sample for training
train = df[~test_bool]

#Print out sample counts
print(str(len(train)) + ' records for training model')
print(str(len(test)) + ' records for testing model')
#Define final variables to be used in the model building process
#The below list indicates the variable name and field label to be
#assigned to it.
model_features = [('age_bins','age'),('education_num_gt_12','ed_lvl'), \
    ('hours_week_gt_40','hr_per_week'), ('cap_gain_gt_0','cap_gain'), \
    ('sex','sex'), ('race','race'),('occupation','occup'), \
    ('marital_status','MS')]

def create_cond_probs(train,test,x):
    variable,label = x
    cond_probs = pd.crosstab(train[variable],train.over_50_bool).apply(lambda r: r/r.sum(),axis=0)
    cond_probs.columns = ['P('+label+'|False)','P('+label+'|True)']
    new_test = test.merge(right=cond_probs,how='inner',left_on=variable,right_index=True,sort=False)
    return new_test

#Assign conditional probabilities to validation data
for var, label in model_features:
    x = (var,label)
    test = create_cond_probs(train,test,x)

#Create percent frequencies for either target class
target_probs = test['over_50_bool'].value_counts(normalize=True).to_dict()

#Add likelihoods for each income category to dataframe
test['True_Likelihood'] = target_probs[True]
test['False_Likelihood'] = target_probs[False]

#Extract conditional probabilities and target likelihoods by target
#for multiplying into each other
under50_cols = [col for col in test.columns if 'False' in col]
over50_cols = [col for col in test.columns if 'True' in col]
test['Under50_Score'] = 1
test['Over50_Score'] = 1

# Multiply conditional probabilities under assumption
# respondent makes less than 50k
for col in under50_cols:
    test['Under50_Score'] = test['Under50_Score'].multiply(test[col],axis=0)

# Multiply conditional probabilities under assumption
# respondent makes over than 50k
for col in over50_cols:
    test['Over50_Score'] = test['Over50_Score'].multiply(test[col],axis=0)

#Get predictions based on final target scores
def get_pred(row):
    if row['Under50_Score'] > row['Over50_Score']:
        return False
    else:
        return True

#Compare predictions to actual target test_dataues
def assess_accuracy(row):
    if row['pred'] == row['over_50_bool']:
        return 'Accurate'
    else:
        return 'Inaccurate'

#Assign a prediction based on training data
test['pred'] = test.apply(get_pred,axis=1)

#Evaluate if prediction was accurate or not
test['accurate'] = test.apply(assess_accuracy,axis=1)

#Print out accuracy on test data
print('Model accuracy for Testing set (' + str(len(test)) + " records):")
print(test['accurate'].value_counts(normalize=True))
