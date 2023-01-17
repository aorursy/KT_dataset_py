%matplotlib inline

import sklearn as sk

import math

import matplotlib as mpl

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import re

import itertools

import scipy

from scipy.interpolate import griddata

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier
#

# Import the test and training data.

#



train_data = pd.read_csv("../input/train.csv",

                        dtype={'Cabin':np.object, 'Ticket':np.object, 'Age':np.float,

                              'SibSP':np.int, 'Parch':np.int, 'Fare':np.float,

                              'Embarked':np.object, 'Sex': np.object}, 

                         na_values={'Cabin':'', 'Ticket':'', 

                                    'Embarked':'', 'Sex':''}, 

                         keep_default_na=False)



# Set the index and create a family size field. Make a single a family size of 1

train_data.set_index('PassengerId', inplace=True)



test_data = pd.read_csv("../input/test.csv",

                        dtype={'Cabin':np.object, 'Ticket':np.object, 'Age':np.float,

                              'SibSP':np.int, 'Parch':np.int, 'Fare':np.float,

                              'Embarked':np.object, 'Sex': np.object}, 

                         na_values={'Cabin':'', 'Ticket':'', 

                                    'Embarked':'', 'Sex':''}, 

                         keep_default_na=False)



# Set the index and create a family size field. Make a single a family size of 1

test_data.set_index('PassengerId', inplace=True)

df_analysis = pd.concat([train_data, test_data], axis=0)

df_analysis['family_size'] = df_analysis['SibSp'] + df_analysis['Parch'] + 1



df_analysis.tail(5)
def extract_title_str(name_str):

    """Extract a title from a name string"""

    mobj = re.search(", (.*?)[\.| ]", name_str)

    mobj_str = (mobj.group()

                .strip(' ,.').

                lower())

    return(mobj_str)

title_ser = df_analysis['Name'].apply(extract_title_str)

title_ser.name ='title'

title_ser.value_counts()
title_grp_ser = title_ser.apply(lambda x: x if x in ['mr', 'mrs', 'miss', 'master'] else 'other')

title_grp_ser.name ='title_grp'

title_grp_ser.value_counts()
df_analysis = pd.concat([df_analysis, title_ser, title_grp_ser], axis=1)



agg_df_title = df_analysis.groupby('title_grp').agg({'Survived':['mean','sem']})

agg_df_title['Survived', 'mean'].plot(kind='bar', yerr=agg_df_title['Survived', 'sem'], alpha = 0.5, error_kw=dict(ecolor='k'));

plt.gcf().suptitle('Survival probability by title');
#

# Examine the cabin/deck variable.  Many are missing

#



def extract_deck_str(cabin_str):

    """Extract a deck letter from a cabin string"""

    return(cabin_str[0:1].upper())



df_analysis['Cabin'].apply(extract_deck_str).value_counts().sort_values()

deck_ser= (df_analysis['Cabin']).apply(extract_deck_str)

deck_ser_mod = deck_ser.apply(lambda x: x if x in ['A','B','C','D','E','F'] 

                              else 'TG' if x in ['T', 'G'] else 'unk')

deck_ser_mod.name = 'deck_mod'



df_analysis = pd.concat([df_analysis, deck_ser_mod], axis=1)

df_analysis.tail()

agg_df_deck = df_analysis.dropna().groupby('deck_mod').agg({'Survived':['mean','sem']})

agg_df_deck['Survived', 'mean'].plot(kind='bar', 

                                     yerr=agg_df_deck['Survived', 'sem'], 

                                     alpha = 0.5, error_kw=dict(ecolor='k'));

plt.gcf().suptitle('Survival probability by deck');



agg_df_deck2 = df_analysis.groupby('deck_mod').agg({'Fare':['mean','sem']})

agg_df_deck2['Fare', 'mean'].plot(kind='bar', 

                                     yerr=agg_df_deck2['Fare', 'sem'], alpha = 0.5, error_kw=dict(ecolor='k'));

plt.gcf().suptitle('Mean fare by deck');

#

# Define some functions for calculating ticket and cabin shares

#



def cabin_share_series(cabin_ser):

    """Returns a series containing the # of people sharing each

    passenger's cabin, when this number can be determined"""

    

    # Get the passenger counts per ticket

    agg_ser = cabin_ser[cabin_ser != ''].value_counts()

    agg_ser.name ='cabin_share'

    

    df_ret = pd.merge(cabin_ser.to_frame(), agg_ser.to_frame(), 

                      how='left', left_on='Cabin', right_index=True,

                      indicator=False)

    

    return(df_ret['cabin_share'])





def clean_ticket(ticket_str):

    """Clean up the ticket string by replacing punctuation, converting

    to upper case, and removing whitespace"""

    ticket_str_clean = re.sub(r'[^\w]', '', ticket_str.strip())

    return (ticket_str_clean)



def ticket_count_frame(ticket_ser):

    """Return a data frame containing 2 series:

        ticket_clean: cleaned-up version of ticket string

        ticket_share_count: number of people with a certain ticket"""

    

    ticket_clean_ser = ticket_ser.apply(clean_ticket)

    ticket_clean_ser.name = 'ticket_clean'

    

    # Get the passenger counts per ticket

    agg_ser = ticket_clean_ser.value_counts()

    agg_ser.name ='ticket_share'



    df_ret = pd.merge(ticket_clean_ser.to_frame(), agg_ser.to_frame(), 

                      how='left', left_on='ticket_clean', right_index=True,

                      indicator=False)

    

    return(df_ret)
#

# Get cabin information from test and training data.

#





share_df = pd.concat([cabin_share_series(df_analysis['Cabin']),

                         ticket_count_frame(df_analysis['Ticket'])], axis=1)





#

# Is the ticket share a good approximation of cabin share?

# Get a grid for ticket vs cabin share, with counts

#



cross_share_df = pd.crosstab(pd.Categorical(share_df['ticket_share']), 

                    pd.Categorical(share_df['cabin_share']))



cross_share_df.index.names= ['Ticket share']

cross_share_df.columns.names=['Cabin share']



cross_share_df.head()

sns.heatmap(cross_share_df, cmap='RdYlGn_r', linewidths=0.5, annot=True,

           cbar_kws={'label': '# of passengers'});

plt.gcf().suptitle('Do people with the same ticket # tend to share a cabin?');



#

# These are not that similar, especially for some tickets

#

#

# Add the ticket share to the frame

#



df_analysis = pd.merge(left=df_analysis, right=share_df[['ticket_share', 

                                                         'cabin_share',

                                                         'ticket_clean']],

                       left_index = True, right_index= True, how='left')

df_analysis.tail(5)

print('null count: ', df_analysis['ticket_share'].isnull().sum())

df_analysis['ticket_share'].value_counts()
#

# Examine the ticket share vs. family size.

# Leave off upper left corner of singletons to get better heat

#



family_ticket_share_df = pd.crosstab(pd.Categorical(df_analysis['ticket_share']), 

                    pd.Categorical(df_analysis['family_size']))

family_ticket_share_df.index.names= ['Ticket share']

family_ticket_share_df.columns.names=['Family size']

family_ticket_share_df.loc[1,1] = np.nan



sns.heatmap(family_ticket_share_df, 

            cmap='RdYlGn_r', linewidths=0.5, annot=True,

                      cbar_kws={'label': '# of passengers'});

plt.gcf().suptitle('Are ticket shares and family size related?');



agg_df = df_analysis.groupby(['family_size', 'ticket_share']).agg({'Survived':['mean','std', 'count']})

agg_df.columns = agg_df.columns.droplevel()

agg_df.head()



# Remove very low-population cells and unstack

agg_df2 = agg_df[agg_df['count'] >= 3]['mean'].unstack(level=1).T



sns.heatmap(agg_df2, cmap='viridis', linewidths=0.5, annot=True,

                      cbar_kws={'label': 'Survival Probability'});

plt.gcf().suptitle('Survival by family size, ticket share');



# View missing values

df_analysis['Embarked'].value_counts().sort_values()
embarked_ser= df_analysis['Embarked'].apply(lambda x: x if x != '' else 'S')

embarked_ser.name = 'embarked_mod'

embarked_ser.value_counts()
# Add the embarcation point to the frame

df_analysis = pd.concat([df_analysis, embarked_ser], axis=1)
print('training null count: ', df_analysis['Fare'].isnull().sum())

print('training zero count: ', df_analysis[df_analysis['Fare']<= 0]['Fare'].count())



print('test null count: ', test_data['Fare'].isnull().sum())

print('test zero count: ', test_data[test_data['Fare']<= 0]['Fare'].count())
sns.boxplot(x='Survived', y='Fare', data = df_analysis);

# Truncate high outliers in plot

plt.ylim([0,140]);
print('null count: ', df_analysis['Age'].isnull().sum())
# Let's look at an age histogram



ax = sns.distplot(df_analysis['Age'].dropna())



# There are many infants aboard.  I wondered if these were legitimate.

# Looking at titles and ages they seem to be.

# There are no 0.0 ages, for example.  Titles tend to be "miss" or 

# "master".  Therefore, this peak seems real


# vector of interesting features

groupby_field_vec= ['SibSp', 'family_size', 'Pclass', 'title_grp', 

                    'Sex', 'Parch', 'embarked_mod', 'deck_mod']



df_list = [df_analysis.groupby(g).agg({'Age':['mean','sem']}) for g in groupby_field_vec]

ind_list = [int('42' + str(x)) for x in range(1,len(groupby_field_vec)+1)]



for i, df in enumerate(df_list):

    df.columns = df.columns.droplevel()

    df['mean'].plot(kind='bar', ax=plt.subplot(ind_list[i]),

                   yerr = df['sem'], figsize=(9,9));



plt.suptitle('Age by various factors')



plt.subplots_adjust(left=None, bottom=0, right=None, top=0.9, wspace=0.3, hspace=0.5)

#

# Look at survival by age

#



df2 = df_analysis.dropna().copy()

df2['age_grp'] = df2['Age'].apply(lambda x: 0 if x < 1 else 1+ int(x/10))



agg_df_age = df2.groupby('age_grp').agg({'Survived':['mean','sem']})

agg_df_age['Survived', 'mean'].plot(kind='bar', yerr=agg_df_age['Survived', 'sem'], alpha = 0.5, error_kw=dict(ecolor='k'));

plt.gcf().suptitle('Survival probability by age');
#

# Function to fill in missing ages

#



def age_filled_ser(age_df):

    """Get a series of inferred/actual passenger ages.  If the age is not NaN,

    it is returned.  Otherwise, the age is inferred using the information in age_df.

       The data frame input must contain the following fields:

        Age

        title_grp

        Pclass

        family_size

        Sex

        Parch

        deck_mod

    The mean ages for the above fields (except Age) will be used to fill in 

    missing data. Some values are thresholded.

       It's possible for a passenger to have a unique combination of items.

    In that case, we do backup fills, removing variables in reverse order

    from the list above (e.g. remove deck, then Sex)

       Assume an appropriate index in the passed-in frame.  

    The output series will be named 'age_mod'"""

    

    field_list= ['title_grp', 'Pclass', 'family_size', 

                       'Sex', 'Parch', 'deck_mod']

    

    fill_df = age_df[['Age'] + field_list].copy()

    

    # Threshold some values

    fill_df['family_size'] = (fill_df['family_size']

                              .apply(lambda x: x if (x <=7) else 7))

    fill_df['Parch'] = (fill_df['Parch']

                              .apply(lambda x: x if (x <=7) else 7))



    

    predictor_list = [field_list[0:x] for x in range(len(field_list), 0, -1)]

    fill_df['age_mod'] = fill_df['Age']

    

    # Fill NAs using descending numbers of predictors

    for predictors in predictor_list:

        fill_df['age_mod'] = (fill_df.groupby(predictors)['age_mod']

                          .transform(lambda x: x.fillna(x.mean())))

    

    return(fill_df['age_mod'])


age_mod_ser = age_filled_ser(df_analysis)

print('null count:', age_mod_ser.isnull().sum())



df_analysis = pd.concat([df_analysis, age_mod_ser], axis=1)
#

# Get mother/nanny and child survival statuses

#



def mother_child_nanny_ticket_frame(ticket_df):

    """Returns 2 data frames:

       1.  child_frame:  A frame consisting of all children, with the child's survival status,

           the mother/nanny's survival status, the survival status of the youngest child, 

           a youngest child flag, and a nanny flag.

       2.  mother_nanny_df: A frame consisting of all mother/nannies, with the survival status,

           the youngest child's survival status, and a nanny flag.

    Data frames are joined by clean ticket number.  """

    

    # Find tickets with children and add a "nanny flag"

    child_tickets_sort = (ticket_df.copy()[ticket_df['age_mod'] < 15]

                          .sort_values(['ticket_clean', 'age_mod', 'Survived', 'Parch'],

                                      ascending=[True, True, False, False]))



    child_tickets_sort['nanny_flag'] = (child_tickets_sort['Parch']

                                        .apply(lambda x: 1 if x ==0  else 0))

    

    # Get the youngest child's survival status and flag

    child_tickets_grp = child_tickets_sort.groupby('ticket_clean')

    youngest_child_df = (child_tickets_grp

                         .head(1)

                         .rename(columns={'Survived': 'survived_youngest'}))[['ticket_clean', 

                                                                              'nanny_flag', 

                                                                              'survived_youngest']]

    # Get the mother/nannny information

    ticket_df_summary = ticket_df.reset_index()

    merge_df = pd.merge(ticket_df_summary, youngest_child_df,

                        how='right', left_on='ticket_clean', 

                        right_on='ticket_clean')

    mother_nanny_merge_df = merge_df[(merge_df['Sex'] == 'female') & (merge_df['age_mod'] >= 18) &

                     ((merge_df['nanny_flag'] == 1) | (merge_df['Parch'] > 0))]

    mother_nanny_df = (mother_nanny_merge_df

                       .sort_values(['ticket_clean', 'age_mod'])

                       .groupby('ticket_clean')

                       .head(1))[['Survived', 'ticket_clean',

                                  'nanny_flag','survived_youngest', 'PassengerId']]

    mother_nanny_df.set_index('PassengerId', inplace=True)

    

    # Get information for all children - include youngest survival and mother's survival



    child_tickets_summary = child_tickets_sort.reset_index()[['PassengerId', 'Survived', 'ticket_clean']]

    child_merge_df = pd.merge(child_tickets_summary,

                              youngest_child_df[['ticket_clean', 'survived_youngest']],

                              how='left', left_on='ticket_clean', right_on='ticket_clean')

    mother_nanny_summary = (mother_nanny_df

                             .rename(columns={'Survived': 'survived_mother_nanny'}))[['ticket_clean',

                                                                                     'nanny_flag', 

                                                                                     'survived_mother_nanny']]

    child_merge_df = pd.merge(child_merge_df, mother_nanny_summary,

                              how='left', left_on='ticket_clean', right_on='ticket_clean')

    

    # Add the youngest child flag

    youngest_id_summary = (youngest_child_df.reset_index())[['PassengerId']]

    child_merge_df = pd.merge(child_merge_df, youngest_id_summary,

                              how='left', left_on='PassengerId', right_on='PassengerId',

                             indicator = True)

    child_merge_df['youngest_flag'] = child_merge_df['_merge'].apply(lambda x: 1 if x == 'both' else 0)

    child_merge_df.set_index('PassengerId', inplace=True)

    child_final_df = child_merge_df[['Survived','ticket_clean', 'youngest_flag', 'nanny_flag', 'survived_youngest',

                                    'survived_mother_nanny']]

    

    return(mother_nanny_df, child_final_df)
(mother_nanny_df, child_df) = mother_child_nanny_ticket_frame(df_analysis[['age_mod', 'ticket_clean', 

                                                                           'Survived', 'Parch', 'Sex']])
pd.crosstab(mother_nanny_df['Survived'], mother_nanny_df['survived_youngest'])
child_older_df = child_df[child_df['youngest_flag'] == 0]

pd.crosstab(child_older_df['Survived'], child_older_df['survived_mother_nanny'])
pd.crosstab(child_older_df['Survived'], child_older_df['survived_youngest'])
mother_nanny_df['mother_nanny_flag'] = 1

child_df['mother_nanny_flag'] = 0

df_analysis_infer_ticket = pd.concat([mother_nanny_df, child_df], axis=0)



df_analysis_infer_ticket.sort_index().head()
def set_inferred_survival(data_row):

    """Takes a row from the ticket inferrence frame and infers a survival status.

    Returns the survival status as a Series"""

    if ((data_row['mother_nanny_flag'] == 0) &

        (not(np.isnan(data_row['survived_mother_nanny'])))):

        return (data_row['survived_mother_nanny'])

    else:

        return (data_row['survived_youngest'])                                
df_analysis_infer_ticket['survived_ticket_inferred'] = (df_analysis_infer_ticket

                                                        .apply(lambda x: set_inferred_survival(x),

                                                       axis=1))  
pd.crosstab(df_analysis_infer_ticket['Survived'], df_analysis_infer_ticket['survived_ticket_inferred'])
#

# For how many rows can this procedure infer survival?

#



len(df_analysis_infer_ticket[np.isnan(df_analysis_infer_ticket['Survived']) &

   ~np.isnan(df_analysis_infer_ticket['survived_ticket_inferred'])])



# 38 isn't a lot of cases...
#

# Add in the inferred survival, and the mother/nanny flag

#



df_analysis = pd.merge(df_analysis, df_analysis_infer_ticket[['survived_ticket_inferred', 'mother_nanny_flag']],

                      how = 'left', left_index = True, right_index=True)

df_analysis['mother_nanny_flag'] = df_analysis['mother_nanny_flag'].fillna(0)

df_analysis.head()
#

# Get functions for binary encoding of categorical variables in our data sets.

# Assume we pass lists of possible values

#



def binary_encode_dict(value_list):

    """Gets a dictionary for mapping values to a list of binary digits

    for binary encoding of categorical variables"""

    

    # Get the # of items in the list and the binary digits required to encode

    max_bin = len(value_list)

    num_bits = math.ceil(math.log(max_bin, 2))



    

    # Get the binary encodings for each level

    str_dict = {value_list[i]: list("{0:b}".format(i).zfill(num_bits))

                for i in range(0, max_bin)}

    

    # Return the bit count and encoding dict

    return (num_bits, str_dict)



def binary_encode_series(name_list, data_series, col_prefix):

    """Apply binary encoding to values in a series, returning a data

    frame consisting of the encoded values, with sequential columns"""

    

    (col_num, col_dict) = binary_encode_dict(name_list)

    col_df = pd.DataFrame(data_series.apply(lambda x: col_dict[x])

                          .values.tolist(),

                          columns=[col_prefix + str(i) for i in range(0,col_num)],

                         index = data_series.index)

    return(col_df)
#

# Write the function that processes the data set, cleaning up fields,

# encoding categorial fields, etc.

#



def df_prepare_analysis_binary(input_df, analysis_df):

    """Get a data frame for analysis using the random forest model.

    Use binary encoding of categorical variables.  Assume some

    predictors have been filled in via the analysis_df object"""

    

    # Get the pre-processed fields for the data of interest

    this_analysis_data = analysis_df.loc[input_df.index]

    

    # Add the ticket share

    ret_df = this_analysis_data[['SibSp', 'Parch', 'Pclass', 'ticket_share', 'deck_mod', 

                                'title_grp', 'family_size', 'age_mod',

                                'Sex', 'embarked_mod', 'Fare', 'survived_ticket_inferred',

                                'mother_nanny_flag']]



    # Fill in any NA fares

    fare_mod_ser = ret_df['Fare'].fillna(np.median(analysis_df['Fare'].dropna()))

    fare_mod_ser.name = 'fare_mod'



    # Binary encoding for categorical fields - Sex

    sex_ser_names = ['male', 'female']

    sex_df = binary_encode_series(sex_ser_names, ret_df['Sex'], 'sex_')

    

    # Binary encoding for categorical fields - Deck (modified)

    deck_ser_names = ['A', 'B', 'C', 'D', 'E', 'F', 'TG', 'unk']

    deck_df = binary_encode_series(deck_ser_names, ret_df['deck_mod'], 'deck_mod_')



    # Binary encoding for categorical fields - Title (modified)

    title_grp_names = ['mr', 'mrs', 'miss', 'master', 'other']

    title_df = binary_encode_series(title_grp_names, ret_df['title_grp'], 'title_grp_')

    

    # Binary encoding for categorical fields - Embarkation point (modified)

    embarked_ser_names = ['S', 'Q', 'C']

    emb_df = binary_encode_series(embarked_ser_names, ret_df['embarked_mod'], 'embarked_mod_')



    ret_df = pd.concat([ret_df, fare_mod_ser, emb_df, sex_df, deck_df, title_df], axis=1)

    return(ret_df)
#

# Create a function for testing the Extra Trees model on the data set.

# This will allow us to vary parameters for the model, including

# the predictor, random state, # of trees, etc.  We will also be

# able to assess the effects of different training/test sets

#  



def test_extra_trees_model(train_test_df, predictor_columns,

                          tt_split_size =0.3, tt_random_state = None,

                          et_n_estimators=10, et_random_state = None,

                          et_max_features = 'auto', et_min_samples_split=2,

                          et_max_leaf_nodes = None):

    '''Test the ExtraTrees model with the training data and certain predictor columns.

    The training data will be split into "training" and "test" portions.  The random state 

    of the split may be passed in as a parameter.  After the split, the ExtraTrees model is run

    and accuracy results obtained.  Extra trees parameters and random states may be passed in as

    parameters also.'''

    

    # Get the test data - split our training set.

    train_test_df_X = train_test_df[predictor_columns]

    train_test_df_Y = train_test_df['Survived']

    X_train, X_test, Y_train, Y_test = train_test_split(train_test_df_X, 

                                                        train_test_df_Y,

                                                        test_size=tt_split_size,

                                                        random_state = tt_random_state)

    

    # Fit the model, and get feature importances

    model = sk.ensemble.ExtraTreesClassifier(n_estimators = et_n_estimators,

                                             random_state = et_random_state,

                                             max_features = et_max_features,

                                             min_samples_split = et_min_samples_split,

                                             max_leaf_nodes = et_max_leaf_nodes)

    fitted_model = model.fit(X_train, Y_train)

    importances = model.feature_importances_

    

    # Process importances into data frame

    importances_dict = {predictor_columns[i]: importances[i] for i in range(0, len(importances))}

    importances_df = (pd.DataFrame.

                  from_dict(importances_dict, orient="index")

                  .rename(columns={0: 'importance'})

                 .sort_values(by='importance', ascending=False))

    

    # Get the predictions, adding in information from the mother/child inferrences

    predictions_raw = fitted_model.predict(X_test)

    predictions_ticket = np.array(train_test_df.loc[X_test.index, 'survived_ticket_inferred'], dtype=pd.Series)

    vfunc = np.vectorize(lambda x,y: y if np.isnan(x) else x)

    predictions = vfunc(predictions_ticket, predictions_raw)

    

    # Get the predictions, accuracy score, and confusion matrics

    confusion_matrix = sk.metrics.confusion_matrix(Y_test,predictions)

    accuracy_score = sk.metrics.accuracy_score(Y_test, predictions)

    

    return (importances_df, confusion_matrix, accuracy_score)

    
#

# Defne the predictor columns

#



predictor_columns = ['Parch', 'fare_mod', 'Pclass', 'family_size', 'ticket_share',

                     'age_mod', 'sex_0', 

                     'deck_mod_0','deck_mod_1', 'deck_mod_2', 

                     'title_grp_0', 'title_grp_1', 'title_grp_2',

                     'embarked_mod_0', 'embarked_mod_1', 'mother_nanny_flag']

#

# Run a simple ExtraTrees test, with default parameters.

# Print the accuracy and confusion matrix

#



train_mod_df = pd.concat([df_prepare_analysis_binary(train_data, df_analysis),

                          train_data['Survived']], axis=1)

(importances_df, confusion_matrix, accuracy_score) = test_extra_trees_model(train_mod_df,

                                                                           predictor_columns)



# print the accuracy info

print('accuracy score: {0:.4g}'.format(accuracy_score))

print('false positives: {0:d}; sensitivity: {1:.3g}'

      .format(confusion_matrix[0,1],

              (confusion_matrix[0,0]/sum(confusion_matrix[0,:]))))

print('false negatives: {0:d}; specificity: {1:.3g}'

      .format(confusion_matrix[1,0],

              (confusion_matrix[1,1]/sum(confusion_matrix[1,:]))))
# Print information about feature importances

importances_df[::-1].plot(kind='barh', legend='False');

plt.xlabel('importance');

plt.title('Variable importances in ExtraTrees model');
#

# Use dictionary comprehension to repeat the model tests.  Use the

# same cross-validation split

#



num_items_test = 101



accuracy_dict_et = {x:(test_extra_trees_model(train_mod_df,

                                        predictor_columns,

                                        tt_random_state = 100,

                                        et_random_state = x))[2] for

                x in range(0, num_items_test)}





accuracy_dict_et_val = list(accuracy_dict_et.values())

plt.hist(accuracy_dict_et_val);

plt.title('Accuracy scores from various model random states');

print('mean accuracy: {}'.format(np.mean(accuracy_dict_et_val)))



# Get the index of the median accuracy, for later random state testing

median_rs = sorted(accuracy_dict_et, key = accuracy_dict_et.__getitem__)[int(num_items_test/2)]

print('median accuracy: {}'.format(accuracy_dict_et[median_rs]))
#

# Let's see how different training slices affect the score.

# Keep the ExtraTrees random state constant for now (use the

# median state selected above)

# Keep the train/test split constant for now (set a random state)

#



num_items_test = 101



accuracy_dict_tt = {x:(test_extra_trees_model(train_mod_df,

                                        predictor_columns,

                                        tt_random_state = x,

                                        et_random_state = median_rs))[2] for

                x in range(0, num_items_test)}





accuracy_dict_tt_val = list(accuracy_dict_tt.values())

plt.hist(accuracy_dict_tt_val);

plt.title('Accuracy scores from various training sets');

print('mean accuracy: {}'.format(np.mean(accuracy_dict_et_val)))
#

# Do multiple ET fits, letting both the test slice and the ET random state float

#



num_items_test = 200



st_pred_items = {x:(test_extra_trees_model(train_mod_df,

                                        predictor_columns))[2] for

                x in range(0, num_items_test)}





predictor_columns_no_ticket_share = [x for x in predictor_columns

                                    if x != 'ticket_share']



ns_pred_items = {x:(test_extra_trees_model(train_mod_df,

                                        predictor_columns_no_ticket_share))[2] for

                x in range(0, num_items_test)}





ticket_compare_df = pd.concat([pd.Series(list(st_pred_items.values()), name='With'),

          pd.Series(list(ns_pred_items.values()), name='Without')], axis=1)



# Create a box plot

sns.boxplot(x=['With', 'Without'], y=[ticket_compare_df['With'], ticket_compare_df['Without']]);

plt.title('Comparison of model results with and without ticket share');

#

# Compare a model with one-hot encoding vs. binary encoding.

#



# Add "one hot" columns to the data frame.

ind_col_list = ['deck_mod', 'Sex', 'title_grp', 'embarked_mod']

one_hot_df = pd.get_dummies(train_mod_df[ind_col_list], 

                            prefix=['deck_hot', 'sex_hot', 'title_hot', 'emb_hot'],

                            columns=ind_col_list)

train_mod_df_hot = pd.concat([train_mod_df, one_hot_df], axis=1)





predictor_columns_hot = ([x for x in predictor_columns

                                    if x not in 

                                     ['sex_0', 'deck_mod_0','deck_mod_1', 

                                      'deck_mod_2', 'title_grp_0', 'title_grp_1', 

                                      'title_grp_2','embarked_mod_0', 'embarked_mod_1']] +

                                     ['sex_hot_female','deck_hot_A','deck_hot_B', 

                                      'deck_hot_C', 'deck_hot_D','deck_hot_E', 'deck_hot_F', 

                                      'deck_hot_TG', 'title_hot_master', 

                                      'title_hot_miss', 'title_hot_mr','title_hot_mrs',

                                       'emb_hot_C', 'emb_hot_Q'])





train_mod_df_hot[predictor_columns_hot].head()



hot_pred_items = {x:(test_extra_trees_model(train_mod_df_hot,

                                        predictor_columns_hot))[2] for

                x in range(0, num_items_test)}



hot_compare_df = pd.concat([pd.Series(list(st_pred_items.values()), name='Binary'),

          pd.Series(list(hot_pred_items.values()), name='OneHot')], axis=1)



sns.boxplot(x=['Binary', 'OneHot'], y=[hot_compare_df['Binary'], hot_compare_df['OneHot']]);

plt.title('Categorical variable encoding effects on model results');



#

# Tune the number of trees.  Do multiple trials per tree count, and 

# try a tree count from 1 to 30

#



tree_count = range(1, 30)

num_trials = 20



st_pred_items = {t: [test_extra_trees_model(train_mod_df,

                                           predictor_columns,

                                           et_n_estimators=t)[2] for

                x in range(0,num_trials)] for t in tree_count}



tree_count_df = pd.DataFrame.from_dict(st_pred_items, orient='index')



test_df = pd.concat([tree_count_df.reset_index()['index'], 

                     tree_count_df.mean(axis=1), 

                     tree_count_df.sem(axis=1)], 

                    axis=1).dropna()

test_df.columns = ['trees', 'mean', 'sem']



test_df.plot.scatter(x='trees', y='mean', yerr='sem');

plt.title('Tree count variation in model results');



#

# Tune the # of samples required for node splits

#



split_count = range(2, 25)

num_trials = 20



sp_pred_items = {s: [test_extra_trees_model(train_mod_df,

                                           predictor_columns,

                                           et_min_samples_split=s)[2] for

                x in range(0,num_trials)] for s in split_count}



split_count_df = pd.DataFrame.from_dict(sp_pred_items, orient='index')



test_df = pd.concat([split_count_df.reset_index()['index'], 

                     split_count_df.mean(axis=1), 

                     split_count_df.sem(axis=1)], 

                    axis=1).dropna()

test_df.columns = ['split samps', 'mean', 'sem']



test_df.plot.scatter(x='split samps', y='mean', yerr='sem');

plt.title('Split sample count effects on model results');

#

# Try different # of features considered at the splits.

# The default is sqrt(n_features).

#



#

# Tune the # of features required for node splits

#



feature_count = range(1, len(predictor_columns))

num_trials = 20



fp_pred_items = {f: [test_extra_trees_model(train_mod_df,

                                           predictor_columns,

                                           et_max_features=f,

                                           et_min_samples_split=10)[2] for

                x in range(0,num_trials)] for f in feature_count}



feature_count_df = pd.DataFrame.from_dict(fp_pred_items, orient='index')



test_df = (pd.concat([feature_count_df.mean(axis=1), 

                     feature_count_df.sem(axis=1)], 

                    axis=1).reset_index()

           .dropna())

test_df.columns = ['features', 'mean', 'sem']



test_df.plot.scatter(x='features', y='mean', yerr='sem');

plt.gcf().suptitle('Model tuning: # features compared to split')

plt.title('Model effects for # features compared at splits');



#

# Try to tune the # of leaf nodes

#



nodes = range(5, 200, 10)

num_trials = 20



st_pred_items = {t: [test_extra_trees_model(train_mod_df,

                                           predictor_columns,

                                           et_n_estimators=10,

                                           et_min_samples_split=15,

                                           et_max_leaf_nodes = t)[2] for

                x in range(0,num_trials)] for t in nodes}



node_count_df = pd.DataFrame.from_dict(st_pred_items, orient='index')



test_df = (pd.concat([node_count_df.mean(axis=1), 

                     node_count_df.sem(axis=1)], 

                    axis=1).reset_index()

           .dropna())

test_df.columns = ['leaf_nodes', 'mean', 'sem']



test_df.plot.scatter(x='leaf_nodes', y='mean', yerr='sem');

plt.title('Leaf node count effects on model results - higher split count');

#

# Try the tree-tune again, but with better split samps

#



tree_count = range(1, 100, 5)

num_trials = 20



st_pred_items = {t: [test_extra_trees_model(train_mod_df,

                                           predictor_columns,

                                           et_n_estimators=t,

                                           et_min_samples_split=15,

                                           et_max_leaf_nodes=25)[2] for

                x in range(0,num_trials)] for t in tree_count}



tree_count_df = pd.DataFrame.from_dict(st_pred_items, orient='index')



test_df = (pd.concat([tree_count_df.mean(axis=1), 

                     tree_count_df.sem(axis=1)], 

                    axis=1).reset_index()

           .dropna())

test_df.columns = ['trees', 'mean', 'sem']



test_df.plot.scatter(x='trees', y='mean', yerr='sem');

plt.title('Tree count effects on model results - higher split count');

#

# Write a function to return predictions for a test set, based

# on a training set

#



def test_extra_trees_model_prediction(train_df, test_df,

                                      predictor_columns,

                                      get_accuracy = False,

                                      et_n_estimators=20, 

                                      et_random_state = None,

                                      et_max_features = 'auto', 

                                      et_min_samples_split=15,

                                      et_max_leaf_nodes=25):

    '''Train an ExtraTrees model, and get a prediction.  Optionlly 

    test the prediction (via an accuracy score).  '''



    # Get the training data

    X_train = train_df[predictor_columns]

    Y_train = train_df['Survived']

    

    # Fit the model

    model = sk.ensemble.ExtraTreesClassifier(n_estimators = et_n_estimators,

                                             random_state = et_random_state,

                                             max_features = et_max_features,

                                             min_samples_split = et_min_samples_split,

                                             max_leaf_nodes = et_max_leaf_nodes)

    fitted_model = model.fit(X_train, Y_train)



    # Get the predictions,

    X_test = test_df[predictor_columns]

    predictions_raw = fitted_model.predict(X_test)

    predictions_ticket = np.array(test_df.loc[X_test.index, 'survived_ticket_inferred'], dtype=pd.Series)

    vfunc = np.vectorize(lambda x,y: y if np.isnan(x) else x)

    predictions = vfunc(predictions_ticket, predictions_raw)

    pred_ser = pd.Series(predictions, index = X_test.index).sort_index()



    # Get an accuracy score if indicated

    if (get_accuracy):

        Y_test = test_df['Survived']

        accuracy_score = sk.metrics.accuracy_score(Y_test, predictions)

    else:

        accuracy_score = None

    

    return (pred_ser, accuracy_score)

    
#

# Test ensembling by splitting the training data again.

# Check that this ensembling brings the overall score near

# the average score for individual runs

#



ens_train, ens_test = train_test_split(train_mod_df, test_size=0.2, random_state = 1)



#

# Repeat modeling for a range of random states

#



num_trials = 21



st_pred_items = [test_extra_trees_model_prediction(ens_train, ens_test, predictor_columns,

                                                    get_accuracy=True,

                                                    et_min_samples_split=15,

                                                    et_random_state = x) 

                 for x in range(0,num_trials)]



predictions = [item[0] for item in st_pred_items]

scores = [item[1] for item in st_pred_items]



#

# Combine predictions - use weighted 

# average majority vote for each observation

#



p_df = pd.DataFrame(predictions).transpose().sort_index()

ens_pred = p_df.sum(axis=1).apply(lambda x: 1 if x > int(num_trials/2) else 0)



ens_Y_test = ens_test['Survived'].sort_index()

accuracy_score = sk.metrics.accuracy_score(ens_Y_test, ens_pred)

print('ensemble accuracy score:{}'.format(accuracy_score))

print('mean score for individual predictions: {}'.format(np.mean(scores)))

print('std dev for individual predictions: {}'.format(np.std(scores)))
#

# Re-train the model using all the available training data.  

# Then apply the model to the test data, including ensembling

#



test_mod_df = df_prepare_analysis_binary(test_data, df_analysis)





#

# Repeat modeling for a range of random states

#



num_trials = 21



fin_pred_items = [test_extra_trees_model_prediction(train_mod_df, test_mod_df, 

                                                   predictor_columns,

                                                   get_accuracy=False,

                                                   et_min_samples_split=15,

                                                   et_random_state = x) 

                 for x in range(0,num_trials)]



predictions = [item[0] for item in fin_pred_items]



#

# Combine predictions - use weighted 

# average majority vote for each observation

#



final_pred_df = pd.DataFrame(predictions).transpose().sort_index()

ens_pred = (final_pred_df.sum(axis=1)

            .apply(lambda x: 1 if x > int(num_trials/2) else 0))

ens_pred.name = 'Survived'

ens_pred.head(10)
#

# Export the predictions to a file

#



ens_pred.to_csv('predictions_vc20171003.csv', header='True')
print('training data survival rate: {0:.3g}'

      .format(train_mod_df['Survived'].mean()))

print('test predicted survival rate: {0:.3g}'

      .format(ens_pred.mean()))
#

# Survival by sex

#



sex_dict = {'train' : train_mod_df[['Sex']], 'test' : test_mod_df[['Sex']]}

sex_comp_df = pd.concat(sex_dict.values(),axis=0,keys=sex_dict.keys())



sur_dict = {'train' : train_mod_df['Survived'], 'test' : ens_pred}

sur_comp_df = pd.DataFrame(pd.concat(sur_dict.values(),axis=0,keys=sur_dict.keys()))



test_df = pd.concat([sex_comp_df, sur_comp_df], axis=1)



t = (test_df.reset_index(0)

     .pivot_table(index='Sex', columns='level_0', aggfunc=[np.mean, scipy.stats.sem]))

t['mean'].plot(kind='bar', yerr=t['sem']);

plt.title('Survival rates by sex: test and training data');

#

# Survival by age.

#



age_dict = {'train' : train_mod_df['age_mod'], 

            'test' : test_mod_df['age_mod']}

age_comp_df = pd.concat(age_dict.values(),axis=0,keys=age_dict.keys())



test_df = pd.concat([age_comp_df, sur_comp_df], axis=1)

test_df['age_grp'] = test_df['age_mod'].apply(lambda x: 20*int(x/20)if x < 60 else 60)



t = (test_df.reset_index(0)

     .pivot_table(index='age_grp', values='Survived',

                  columns='level_0', aggfunc=[np.mean, scipy.stats.sem]))

t.head()

t['mean'].plot(kind='bar', yerr=t['sem']);

plt.title('Survival rates by age grp: test and training data');
ax = sns.distplot(test_mod_df['age_mod'])