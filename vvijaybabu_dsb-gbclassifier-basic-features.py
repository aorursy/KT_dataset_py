import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import json

from pandas.io.json import json_normalize



import os

directory = "/kaggle/input/data-science-bowl-2019/"

outdir = ""

for dirname, _, filenames in os.walk(directory):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#load and analyze the specs data

specs = pd.read_csv(directory +"specs.csv")  

print("no of rows", specs.count())

specs.head()
#load and analyze the test data

test = pd.read_csv(directory + "test.csv")

print(test.count(axis=0))

test.head()



#load and analyze the submission data

submission = pd.read_csv(directory + "sample_submission.csv")

print(submission.count(axis=0))

submission.head()
#in the training set, learn more about data

#load the full train.csv



train = pd.read_csv(directory +"train.csv")

print(len(train.index))

train.head()



#load and analyze the labels data

train_labels = pd.read_csv(directory + "train_labels.csv")

print(train_labels.count())

train_labels.head()



train.nunique()
test.nunique()
#we will create  a function that will build all features

#this will be used for train and test data

#the caller of this function shall pass the basic feature rows and the dataset from which to extract features

def create_features(features,dataset):

    ftr = dataset[(dataset.type == 'Assessment')&(dataset.event_code == 2000)] # all game_session start rows

    ftr_1 = dataset[(dataset.type == 'Assessment')&((dataset.event_code == 4100)|(dataset.event_code == 4110))]

    ftr_1 = ftr_1[((ftr_1.event_code == 4100) & (ftr_1.title != 'Bird Measurer (Assessment)')) | 

                             ((ftr_1.event_code == 4110) & (ftr_1.title == 'Bird Measurer (Assessment)'))]

    #collected all assessment rows

    ftr_subset = ftr.loc[:,('installation_id','game_session','event_code')]

    ftr_1_subset = ftr_1.loc[:,('installation_id','game_session','event_code')]

    ftr_1_subset.reset_index(inplace=True)

    ftr_1_subset.set_index(['installation_id','game_session'],inplace=True)

    ftr_subset = ftr_subset.join(ftr_1_subset,on=('installation_id','game_session'),rsuffix='_4100')

    ftr_subset= ftr_subset[ftr_subset['event_code_4100'].isna()==False]

    ftr_subset.drop(columns=['index','event_code_4100'],inplace=True)

    gpby = ftr_subset.groupby(['installation_id','game_session']).count()

    gpby = gpby.groupby(['installation_id']).count()

    features = features.join(gpby,on='installation_id',rsuffix='_with4100')

    features.rename(columns={'event_code_with4100':'total_2000_w_4100'},inplace=True)

    #if NaN is in total_2000_w_4100, then it means that there is no attempt

    features.loc[:,'total_2000_w_4100'].fillna(0,inplace=True)



    gpby = ftr_subset.groupby(['installation_id']).count()

    features = features.join(gpby,on='installation_id',rsuffix='_withall4100')

    features.rename(columns={'event_code_withall4100':'total_4100'},inplace=True)

    features.drop(columns=['game_session_withall4100'],inplace=True)

    features.loc[:,'total_4100'].fillna(0,inplace=True)



    #now find the gametime count (since gametime is cum time from start)

    subset = dataset.loc[:,('installation_id','title','game_time')]

    addl_ftr = subset.groupby(['installation_id','title']).count()

    addl_ftr.reset_index(inplace=True)

    addl_ftr.set_index(['installation_id','title'],inplace=True)

    addl_ftr = addl_ftr.unstack(1)

    features=features.join(addl_ftr,on=('installation_id'),rsuffix='_addl')



    features = pd.get_dummies(features,prefix='cat',columns=['world','title'])

    features.reset_index(inplace=True)

    

    features.drop(columns=['event_id','timestamp',

                                'event_data','event_count','event_code',

                                'game_time','type',

                                'cat_CRYSTALCAVES','cat_MAGMAPEAK',

                                'cat_TREETOPCITY',],inplace=True)

    features.fillna(0,inplace=True)



    #lets build the count of  accuracy_group for each install_id+title

    #build the temp from train_filtered, and fill "correct" from "event_data"



    temp = dataset[(dataset.type == 'Assessment') & ((dataset.event_code == 4100) | (dataset.event_code == 4110))]

    parsed_df = pd.concat([json_normalize(json.loads(js)) for js in temp['event_data']])

    temp.loc[:,'correct'] = parsed_df['correct'].to_numpy()



    #remove temp rows with 4100 and Bird Measurer

    temp = temp[((temp.event_code == 4100) & (temp.title != 'Bird Measurer (Assessment)')) | 

                             ((temp.event_code == 4110) & (temp.title == 'Bird Measurer (Assessment)'))]



    temp_subset = temp.loc[:,('installation_id','title','game_session','correct','type')]

    temp_gpby = temp_subset.groupby(['installation_id','title','game_session','correct'],observed=True).count()

    temp_gpby = temp_gpby.unstack(-1)



    temp_gpby.fillna(0,inplace=True)

    temp_gpby['accuracy']=temp_gpby[('type',True)]/(temp_gpby[('type',False)]+temp_gpby[('type',True)])

    temp_gpby.drop('type',axis=1,level=0,inplace=True)

    temp_gpby.columns = temp_gpby.columns.get_level_values(0)



    temp_gpby.head()

    temp_gpby =pd.get_dummies(temp_gpby,columns=['accuracy'])

    temp_gpby_copy = temp_gpby.copy(deep=True)

    length = len(temp_gpby.columns)-3

    for i in range(length):

        temp_gpby.drop(temp_gpby.columns[1],axis=1,inplace=True)

    temp_gpby_copy.drop(temp_gpby_copy.columns[[0,length+1,length+2]],axis=1,inplace=True)    

    temp_gpby['accuracy_grp1'] = temp_gpby_copy.sum(axis=1)

    temp_gpby=temp_gpby.groupby(['installation_id','title']).sum()

    temp_gpby = temp_gpby.unstack(-1)

    temp_gpby.columns = [' '.join(col).strip() for col in temp_gpby.columns.values]

    temp_gpby.fillna(-1,inplace=True)

    #now join train_features to temp_gpby

    features = features.join(temp_gpby,on=['installation_id'],rsuffix='_more')

    features.reset_index(inplace=True)

    features.drop(columns=['index','level_0'],inplace=True)

    features.fillna(-1,inplace=True)

    return features





#now prep the train and test data to call create_features



#train data

train['timestamp'] =pd.to_datetime(train['timestamp'])

temp = train[(train.type == 'Assessment') & ((train.event_code == 4100) | (train.event_code == 4110))]

train_filtered_2 = temp[((temp.event_code == 4100) & (temp.title != 'Bird Measurer (Assessment)')) | 

                         ((temp.event_code == 4110) & (temp.title == 'Bird Measurer (Assessment)'))]

#this has only 4100s, now find last 4100

train_filtered_2_subset = train_filtered_2.loc[:,('installation_id','timestamp')]

train_filtered_2_gpby = train_filtered_2_subset.groupby(['installation_id']).max()

train_filtered_2_gpby.reset_index(inplace=True)

train_filtered_2_gpby.set_index('installation_id',inplace=True)

train_filtered_3 = train_filtered_2.join(train_filtered_2_gpby,on=('installation_id'),rsuffix='_max')

train_filtered_3 = train_filtered_3[train_filtered_3.timestamp == train_filtered_3.timestamp_max]

train_filtered_3_subset = train_filtered_3.loc[:,('installation_id','game_session')]

#now find the 2000 with  this game_session and get its timestamp;

train_filtered_4 = train[(train.event_code== 2000)&(train.type=='Assessment')]

train_filtered_4.set_index(['installation_id','game_session'],inplace=True)

train_features = train_filtered_3_subset.join(train_filtered_4,on=('installation_id','game_session'),rsuffix='_r')

train_features.set_index(['installation_id'],inplace=True)

train_filtered = train.join(train_features,on=['installation_id'],rsuffix="_max")

#now remove all NaN rows in col 'timestamp_max' since they are not in train_features

train_filtered = train_filtered[train_filtered['timestamp_max'].isna()== False]

#and remove all rows with timestamp> timestamp_max

train_filtered = train_filtered[train_filtered.timestamp <= train_filtered.timestamp_max]

#now train_filtered has all rows of interest matching test dataset



train_features.reset_index(inplace=True)

train_features = create_features(train_features,train_filtered)



#now build the train_y from train_features and train_labels

train_y_subset = train_features.loc[:,('installation_id','game_session','event_code')]

train_y_subset2 = train_labels.loc[:,('installation_id','game_session','accuracy_group')]

train_y_subset2.set_index(['installation_id','game_session'],inplace=True)

train_y = train_y_subset.join(train_y_subset2,on=['installation_id','game_session'],rsuffix='_r')

train_inst_id=train_features['installation_id']



#now for the test data

temp_test = test[(test.type == 'Assessment') & ((test.event_code == 4100) | (test.event_code == 4110))]



temp_test = temp_test[((temp_test.event_code == 4100) & (temp_test.title != 'Bird Measurer (Assessment)')) | 

                         ((temp_test.event_code == 4110) & (temp_test.title == 'Bird Measurer (Assessment)'))]

temp_test['timestamp'] =pd.to_datetime(temp_test['timestamp'])

#temp_test has all 4100 events



#in test data, find all last 2000 - these are the feature rows

test['timestamp'] =pd.to_datetime(test['timestamp'])

test_ftr = test[(test.type == 'Assessment') & (test.event_code == 2000)]



#now take only the max timestamp for each inst_id 

test_ftr_subset = test_ftr.loc[:,('installation_id','timestamp')]

test_gpby = test_ftr_subset.groupby(['installation_id']).max()

test_gpby.reset_index(inplace=True)

test_gpby.set_index(['installation_id'],inplace=True)

test_features = test_ftr.join(test_gpby,on=('installation_id'),rsuffix='_max')

test_features =test_features.loc[test_features['timestamp']== test_features['timestamp_max']]

test_features.drop(columns=['timestamp_max'],inplace=True)



test_features =create_features(test_features,test)





test_features.to_csv(outdir+'test_features.csv')

train_features.to_csv(outdir+'train_features.csv')

train_y.to_csv(outdir+'train_y.csv')

test_inst_id = test_features['installation_id']

train_features.columns
#now remove the unrequired columns

train_features.drop(columns=['installation_id','game_session','total_2000_w_4100',

                                         'total_4100'],inplace=True)

test_features.drop(columns=['installation_id','game_session','total_2000_w_4100',

                                         'total_4100'],inplace=True)

train_y.drop(columns=['installation_id','game_session','event_code'],inplace=True)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV

from sklearn.metrics import roc_auc_score, confusion_matrix, cohen_kappa_score, make_scorer

from sklearn.ensemble import GradientBoostingClassifier



gbc_params = {

    "learning_rate":[0.01,0.05,0.1,0.2,0.5],

    "n_estimators":[128,256],

    "max_leaf_nodes":[2,4],

    "random_state":[0]    

}



cks = make_scorer(cohen_kappa_score,weights="quadratic")

GBclf = GradientBoostingClassifier()

GScv = GridSearchCV(GBclf,param_grid = gbc_params,cv=5,scoring=cks)

GScv.fit(train_features,train_y.iloc[:,0])

GScv.best_params_
#now score the test set and submit

result = GScv.predict(test_features)

results = pd.DataFrame(data=result)

results['installation_id']=test_inst_id.values

results.rename(columns={0:'accuracy_group'},inplace=True)

results.to_csv(outdir+'submission.csv',index=False)

results.head()