import pandas as pd

import numpy as np

import re

import itertools as it
train=pd.read_json('../input/train.json')

train['listing_id']=train['listing_id'].apply(str)
feature_total=[]

train['features'].apply(lambda x: feature_total.append(x))

feature_total=list(it.chain.from_iterable(feature_total))

len(feature_total)
uniq_feature_total=set(feature_total)

len(uniq_feature_total)
list(uniq_feature_total)[:10]
def feature_star_sep(feature_list):

    '''

    Seperate feature text with * or • as separator

    '''

    new_list=[]

    for feature in feature_list:

        if feature[:2]=='**':

            new=feature[3:-3]

            new_list+new.split(" * ")

        elif feature[:1]=='•':

            new=feature[2:]

            new_list+new.split(" • ")            

        else:

            new_list.append(feature)

            

    return new_list
train['features']=train['features'].apply(feature_star_sep)
from sklearn.feature_extraction.text import CountVectorizer
## Code copied from @sudalairajkumar 

vec=CountVectorizer(stop_words='english', max_features=200)

train['features_new'] = train["features"].apply(lambda y: " ".join(["_".join(x.split(" ")).lower() for x in y]))

tr_sparse = vec.fit_transform(train["features_new"])

feature_names=vec.get_feature_names()
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import log_loss
target_num_map = {'high':0, 'medium':1, 'low':2}

features=tr_sparse.toarray()

labels=train['interest_level'].apply(lambda x: target_num_map[x]).as_matrix()
clf=DecisionTreeClassifier(max_depth=5)
cv=StratifiedShuffleSplit(n_splits=3, test_size=0.3)



for train_idx, test_idx in cv.split(features, labels): 

    features_train,labels_train = features[train_idx],labels[train_idx]

    features_test,labels_test = features[test_idx],labels[test_idx]

    clf.fit(features_train,labels_train)

    print("log loss:",(-1)*round(log_loss(labels_test,clf.predict_proba(features_test)),3))

    

    ## Print out features with high importance

    print('high importance features:')

    for idx in np.where(clf.feature_importances_>0.05)[0]:

        print("  ",feature_names[idx],round(clf.feature_importances_[idx],3))

        
feature_total=[]

train['features'].apply(lambda x: feature_total.append(x))

feature_total=list(it.chain.from_iterable(feature_total))

uniq_feature_total=list(set(feature_total))
def re_search(key):

    '''

    Present all features with specific re pattern

    '''

    result=[]

    my_reg=r""+key

    for item in uniq_feature_total:

        if re.compile(my_reg ,re.IGNORECASE).search(item)!=None:

            result.append(item)

    return result
# Check all text including 'hardwood'

re_search('hardwood')
# Check all text including 'doorman'

re_search('doorman')
# Check all text including 'fee'

re_search('fee')
# Extract no fee

re_search('no\s*\w*\s*fee')
# Extract low fee

re_search('reduce|low\sfee')
# Check all text including 'laundry'

re_search('laundry')
# Extract war and exclude other keyword with 'war' such as warmth and wardrobe

re_search('war\Z|war\s')
# Check all text including 'fitness' or 'gym'

re_search('fitness|gym')
def add_feature(row):

    if re.search(r'hardwood',row['features_new'],re.IGNORECASE)!=None:

        row['hardwood']=1

    else:

        row['hardwood']=0

        

    if re.search(r'doorman',row['features_new'],re.IGNORECASE)!=None:

        row['doorman']=1

    else:

        row['doorman']=0

    

    if re.search(r'no\w*fee',row['features_new'],re.IGNORECASE)!=None:

        row['no_fee']=1

    else:

        row['no_fee']=0

    

    if re.search(r'reduce|low\wfee',row['features_new'],re.IGNORECASE)!=None:

        row['reduce_fee']=1

    else:

        row['reduce_fee']=0



    if re.search(r'laundry',row['features_new'],re.IGNORECASE)!=None:

        row['laundry']=1

    else:

        row['laundry']=0



    if re.search(r'war\Z|war\s|war_',row['features_new'],re.IGNORECASE)!=None:

        row['war']=1

    else:

        row['war']=0



    if re.search(r'fitness|gym',row['features_new'],re.IGNORECASE)!=None:

        row['gym']=1

    else:

        row['gym']=0

        

    return row
train=train.apply(add_feature,axis=1)
train[['hardwood','doorman','no_fee','reduce_fee','laundry','war','gym']].apply(sum)