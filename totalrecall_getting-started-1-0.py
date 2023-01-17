# Imports




# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn, sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
sns.set_style('whitegrid')
##%matplotlib inline



#### import the data
train_users   = pd.read_csv('../input/train_users_2.csv')
test_users    = pd.read_csv('../input/test_users.csv')
gender = pd.read_csv('../input/age_gender_bkts.csv')
sessions = pd.read_csv('../input/sessions.csv')
countries = pd.read_csv('../input/countries.csv')

##all_users = pd.concat((train_users, test_users), axis=0, ignore_index=True)

mobile_browsers = []
for x in train_users['first_browser'].unique():
    if 'Mobile' in x:
        mobile_browsers.append(x)
    else:
        pass 

major_browsers = ['IE', 'Safari', '-unknown- ', 'Chrome', 'Firefox', 'Mobile']  

### group up those first_browsers
train_users['first_browser_grouped'] = np.where(train_users['first_browser'].isin(mobile_browsers), 'Mobile', train_users['first_browser'])
train_users['first_browser_grouped'] = np.where(train_users['first_browser_grouped'].isin(major_browsers), train_users['first_browser_grouped'], 'Other')

### find year of account creation
#train_users['year_account_creation'] = pd.DatetimeIndex(train_users['date_account_created']).year

### group up the first_device_type
dict_first_device_type = {"Mac Desktop": "Desktop",
                          "Windows Desktop": "Desktop",
                          "Desktop (Other)": "Desktop",
                          "iPhone": "Phone/Pad",
                          "iPad": "Phone/Pad",
                          "Android Tablet": "Phone/Pad", 
                          "Android Phone": "Phone/Pad",
                          "SmartPhone (Other)": "Phone/Pad"}
train_users = train_users.replace({"first_device_type": dict_first_device_type})



######### apply the above adjustments to the test dataset
test_users['first_browser_grouped'] = np.where(test_users['first_browser'].isin(mobile_browsers), 'Mobile', test_users['first_browser'])
test_users['first_browser_grouped'] = np.where(test_users['first_browser_grouped'].isin(major_browsers), test_users['first_browser_grouped'], 'Other')
#test_users['year_account_creation'] = pd.DatetimeIndex(test_users['date_account_created']).year
test_users = test_users.replace({"first_device_type": dict_first_device_type})


language_distance = {'language' : ['en', 'du', 'fr', 'es'],
                     'levenshtein_distance_from_en' : [0, 72.61, 92.06, 92.25]}

language_distance = pd.DataFrame(language_distance)

train_users = pd.merge(train_users, language_distance, on = 'language', how = 'left')
test_users = pd.merge(test_users, language_distance, on = 'language', how = 'left')



########## fill in the missing values
train_users['levenshtein_distance_from_en'].fillna(-1)
test_users['levenshtein_distance_from_en'].fillna(-1)
##train_users['year_account_creation'] = pd.DatetimeIndex(train_users['date_account_created']).year
train_users['timestamp_first_active'] = train_users['timestamp_first_active'].astype(str)

train_users['date_account_created'] = pd.to_datetime(train_users['date_account_created'])

#### converting the first active day to a date-time var
train_users['timestamp_first_active_day'] = train_users['timestamp_first_active'].str[:8]
train_users['timestamp_first_active_day'] = pd.to_datetime(train_users['timestamp_first_active_day'], format='%Y%m%d')

#### find the first active year
train_users['timestamp_first_active_year'] = train_users['timestamp_first_active'].str[:4]
train_users['timestamp_first_active_hour'] = train_users['timestamp_first_active'].str[8:10]

#### create a var to see if they searched before joining
#train_users['searched_before_joining'] = (train_users['timestamp_first_active_day'] < train_users['date_account_created'])
#train_users['searched_before_joining'] = train_users['searched_before_joining'] * 1

#### did they do a previous trip? This appears to be a weird variable..
##train_users['first_trip'] = pd.isnull(train_users['date_first_booking']) * 1

major_languages = ['en']  
train_users['language_bucket'] = np.where(train_users['language'].isin(major_languages), 'en', 'other')

##### group up the age variable
labels = [1, 2, 3, 4, 5, 6, 7]
bins = [0, 20, 30, 40, 50, 60, 9000, 100000]
train_users['age'].fillna(10000)
train_users['age_group'] = pd.cut(train_users['age'], bins, right=False, labels=labels)
train_users['age_group'] = train_users['age_group'] * 1

train_users["signup_combo"] = train_users["signup_method"].map(str) + train_users["signup_flow"].map(str)

##### let's group the affiliate_provider variable

major_affiliate_providers = ['direct', 'google', 'bing', 'craigslist', 'facebook']
train_users['affiliate_provider_grp'] = np.where(train_users['affiliate_provider'].isin(major_affiliate_providers), train_users['affiliate_provider'], 'other')
train_users["affiliate_combined"] = train_users["affiliate_provider_grp"].map(str) + train_users["affiliate_channel"].map(str)






###### adjust test so it matches the adjustments made to the train dataset
test_users['timestamp_first_active'] = test_users['timestamp_first_active'].astype(str)
test_users['date_account_created'] = pd.to_datetime(test_users['date_account_created'])
test_users['timestamp_first_active_day'] = test_users['timestamp_first_active'].str[:8]
test_users['timestamp_first_active_day'] = pd.to_datetime(test_users['timestamp_first_active_day'], format='%Y%m%d')
test_users['timestamp_first_active_year'] = test_users['timestamp_first_active'].str[:4]
#test_users['searched_before_joining'] = (test_users['timestamp_first_active_day'] < test_users['date_account_created'])
#test_users['searched_before_joining'] = test_users['searched_before_joining'] * 1
##test_users['first_trip'] = pd.isnull(test_users['date_first_booking']) * 1
test_users['language_bucket'] = np.where(test_users['language'].isin(major_languages), 'en', 'other')
test_users['age'].fillna(10000)
test_users['age_group'] = pd.cut(test_users['age'], bins, right=False, labels=labels)
test_users['age_group'] = test_users['age_group'] * 1
test_users['timestamp_first_active_day'] = pd.to_datetime(test_users['timestamp_first_active_day'], format='%Y%m%d')
test_users["signup_combo"] = test_users["signup_method"].map(str) + test_users["signup_flow"].map(str)
test_users['timestamp_first_active_hour'] = test_users['timestamp_first_active'].str[8:10]
test_users['affiliate_provider_grp'] = np.where(test_users['affiliate_provider'].isin(major_affiliate_providers), test_users['affiliate_provider'], 'other')
test_users["affiliate_combined"] = test_users["affiliate_provider_grp"].map(str) + test_users["affiliate_channel"].map(str)


##train_users.head()
##train_users[train_users['first_browser_grouped'] == 'Mobile']

#### language doesn't appear that helpful.. anyway we can adjust it some?

#train_users.head()

#test = [0]
#train_users['tester'] = np.where((train_users['timestamp_first_active_day'] - train_users['date_account_created']).isin(test), 0, 1)

#fig, (axis1) = plt.subplots(1,1,figsize=(15,5))
#fig, (axis1, axis2, axis3) = plt.subplots(3,1,figsize=(15,15))
#sns.countplot(x='language_bucket', hue = 'country_destination', data=train_users, palette="husl", ax=axis1)
#sns.countplot(x = 'affiliate_channel', hue = 'country_destination', data = train_users, palette = 'husl', ax = axis2)
#sns.countplot(x = 'affiliate_provider_grp', hue = 'country_destination', data = train_users, palette = 'husl', ax = axis3)
#sns.countplot(x = 'signup_app', hue = 'country_destination', data = train_users, palette = 'husl', ax = axis4)
#sns.countplot(x = 'affiliate_provider', hue = 'country_destination', data = train_users[train_users['affiliate_provider'] != 'direct'], palette = 'husl', ax = axis5)
#sns.countplot(x = 'tester', hue = 'country_destination', data = train_users[train_users['tester'] == 1], palette = 'husl', ax = axis1)


#fig, (axis1, axis2) = plt.subplots(2,1,figsize=(15,10))
#sns.countplot(x='affiliate_channel', hue = 'country_destination', data=train_users[train_users['affiliate_provider_grp'] == 'google'], palette="husl", ax=axis1)
#sns.countplot(x='affiliate_provider_grp', hue = 'country_destination', data=train_users[train_users['affiliate_channel'] == 'seo'], palette="husl", ax=axis2)




##### I'm curious, what's the interaction between date_account_created and timestamp_first_active? 
#train_users['date_account_created'] = pd.to_datetime(train_users['date_account_created'])
#train_users['dif_btwn_creation_and_search'] = (train_users['timestamp_first_active_day'] - train_users['date_account_created']).astype('timedelta64[M]')
#train_users['dif_btwn_creation_and_search_rounded'] = train_users['dif_btwn_creation_and_search'].round(-1)


#fig, (axis1) = plt.subplots(1,1,figsize=(15,5))
#sns.countplot(x='dif_btwn_creation_and_search_rounded', hue = 'country_destination', data=train_users, palette="husl", ax=axis1)

#train_users.head()


#fig, (axis1) = plt.subplots(1,1,figsize=(15,5))
#sns.countplot(x='age_group', hue = 'country_destination', data=train_users, palette="husl", ax=axis1)

##### is it worthwhile to group up some of these X vars w/ a lot of subclasses? 




#######

train_users['month_created'] = train_users['date_account_created'].map(lambda x: x.month)
train_users['year_created'] = train_users['date_account_created'].map(lambda x: x.year)
train_users['month_year_created'] = train_users['date_account_created'].map(lambda x: x.year * 1000 + x.month)

train_users['month_first_active'] = train_users['timestamp_first_active_day'].map(lambda x: x.month)
train_users['year_first_active'] = train_users['timestamp_first_active_day'].map(lambda x: x.year)
train_users['month_year_first_active'] = train_users['timestamp_first_active_day'].map(lambda x: x.year * 1000 + x.month)

########
test_users['month_created'] = test_users['date_account_created'].map(lambda x: x.month)
test_users['year_created'] = test_users['date_account_created'].map(lambda x: x.year)
test_users['month_year_created'] = test_users['date_account_created'].map(lambda x: x.year * 1000 + x.month)

test_users['month_first_active'] = test_users['timestamp_first_active_day'].map(lambda x: x.month)
test_users['year_first_active'] = test_users['timestamp_first_active_day'].map(lambda x: x.year)
test_users['month_year_first_active'] = test_users['timestamp_first_active_day'].map(lambda x: x.year * 1000 + x.month)





############
train_users = train_users.drop(['date_account_created', 'timestamp_first_active', 'timestamp_first_active_day', 'date_first_booking'], axis = 1)
test_users = test_users.drop(['date_account_created', 'timestamp_first_active', 'timestamp_first_active_day', 'date_first_booking'], axis = 1)
##### this removes the missing values
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
train_users_imputed = DataFrameImputer().fit_transform(train_users)
test_users_imputed = DataFrameImputer().fit_transform(test_users)


### this will transfer the categorical variables to floats for the algo
def do_treatment(df):
    for col in df:
        if df[col].dtype == np.dtype('O') and df[col].name != 'id' and df[col].name != 'country_destination' and df[col].name != 'age_group' and df[col].name != 'timestamp_first_active_day':
            df[col] = df[col].apply(lambda x : hash(str(x)))

    
do_treatment(train_users_imputed)
do_treatment(test_users_imputed)


#np.any(np.isnan(train_users['id']))
#print(np.all(np.isfinite(col)))

#np.isnan(train_users.any())
#np.isfinite(train_users.any())

#np.isnan(test_users.any())
#np.isfinite(test_users.any())




#train_users.head()


X_train = train_users_imputed.drop(['signup_app', 'affiliate_provider', 'affiliate_channel', 'levenshtein_distance_from_en', 'month_year_first_active', 'month_year_created', 'year_first_active', 'timestamp_first_active_year', 'country_destination', 'id', 'first_browser', 'age', 'language'], axis=1)
y_train = train_users_imputed['country_destination']
X_test = test_users_imputed.drop(['signup_app', 'affiliate_provider', 'affiliate_channel', 'levenshtein_distance_from_en', 'month_year_first_active', 'month_year_created', 'year_first_active', 'timestamp_first_active_year', 'id', 'age', 'first_browser', 'language'], axis = 1)


country_num_dic = {'NDF': 0, 'US': 1, 'other': 2, 'FR': 3, 'IT': 4, 'GB': 5, 'ES': 6, 'CA': 7, 'DE': 8, 'NL': 9, 'AU': 10, 'PT': 11}
num_country_dic = {y:x for x,y in country_num_dic.items()}

y_train    = y_train.map(country_num_dic)

##### build the model

clf = RandomForestClassifier(n_estimators = 200, max_features = 'sqrt',
                             max_depth = None, verbose = 1, n_jobs = -1)
clf.fit(X_train, y_train)
clf_probs = clf.predict_proba(X_test) ## prob of being in a certain class
test_preds = clf.predict(X_test)
test_preds = test_preds.astype(int)

# change values back to original country symbols
test_preds = Series(test_preds).map(num_country_dic)

output = pd.DataFrame(test_users.id).join(pd.DataFrame(clf_probs))
output_melted = pd.melt(output, id_vars = 'id')
# convert type to integer
output_melted['variable'] = output_melted['variable'].astype(int)

# change values back to original country symbols
output_melted['variable'] = Series(output_melted['variable']).map(num_country_dic)

output_sorted = output_melted.sort(['id', 'value'], ascending=[1, 0])
top_5_records = output_sorted.groupby('id').head(5)
top_5_records_trimmed = top_5_records.drop(['value'], axis = 1)
top_5_records_trimmed.columns = ['id', 'country']

final_output = DataFrame(columns=['id', 'country'])
final_output = final_output.append(top_5_records_trimmed)




##### get the output ready for a csv submission
#output = pd.DataFrame(test_users.id).join(pd.DataFrame(test_preds))
#output.columns = ['id', 'country_destination']








###### look at variable importance in the model 

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in indices:
    print(X_train.columns[f], importances[f])
    #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

#for x in range(X_train.shape[1]):
#    print(X_train.columns[x])
### alright, if none of the entries for an id is NDF, then set the 5th obs == NDF
no_ndf_ppl = top_5_records[~top_5_records['id'].isin(top_5_records['id'][top_5_records['variable'] == 'NDF'])]
ndf_ppl = top_5_records[top_5_records['id'].isin(top_5_records['id'][top_5_records['variable'] == 'NDF'])]

no_ndf_ppl_first = no_ndf_ppl.sort(['value'], ascending=[1])
no_ndf_ppl_first_ndf = no_ndf_ppl_first.groupby('id').head(1)
no_ndf_ppl_first_ndf['variable'] = 'NDF'

no_ndf_ppl_first_4 = no_ndf_ppl.sort(['value'], ascending=[0])
no_ndf_ppl_first_other = no_ndf_ppl_first_4.groupby('id').head(4)

##### combine all of the dataframes together
result = pd.concat([no_ndf_ppl_first_ndf, no_ndf_ppl_first_other , ndf_ppl])
result = result.drop(['value'], axis = 1)
result.columns = ['id', 'country']

#### create the final output dataframe
final_output_adjusted = DataFrame(columns=['id', 'country'])
final_output_adjusted = final_output_adjusted.append(result)

#### convert to csv
final_output_adjusted.to_csv('adjusted.csv', index = False, header = ['id', 'country'])



