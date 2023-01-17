# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for visualization

import seaborn as sns

from sklearn import linear_model

import re



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier, plot_importance



%matplotlib inline

### Seaborn style

sns.set_style("whitegrid")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



trainingData = pd.read_json('../input/train.json')





#trainingData['building_id'] = trainingData['building_id'].to_string



trainingData.info()
trainingData.isnull().sum()
sns.countplot(trainingData.interest_level, order=['low', 'medium', 'high']);

plt.xlabel('Interest Level');

plt.ylabel('Number of occurrences');


trainingData['numPics'] = trainingData['photos'].apply(len)



trainingData.info()
features = [x for sublist in trainingData['features'] for x in sublist]



for x in features:

    if "*" in x: features.remove(x)



features = set(features)



features.discard('<null>')



has_ac = [ s for s in features if any(ac_name in s for ac_name in ['A/C', "AC", "Air Conditioning"] ) and not(any(wrong in s for wrong in ["FIRE",'ACT','APT', 'SPACE','YARD'])) ]



trainingData['has_ac'] = [any(ac in feature for ac in has_ac) for feature in trainingData['features'] ]

  

free_included = [ s for s in features if any(ac_name in s for ac_name in ["free","FREE","Free", "Gift", "gift", '1/2 Month fee', "included", "INCLUDED","Included"] ) ]

    

trainingData["included_offer"] = [any(free in feature for free in free_included) for feature in trainingData['features']]



doorman = [ s for s in features if any(ac_name in s for ac_name in ["doorman","DOORMAN","Doorman",'doormen','Doormen', 'full-service', 'concierge','Concierge','Attended Lobby', 'Attended lobby', 'attended lobby'] ) ]

   

trainingData["concierge"] = [any(door in feature for door in doorman) for feature in trainingData['features']]



Washer = [ s for s in features if any(ac_name in s for ac_name in ['Washer', "Dryer",'Washer','Dryer','washer','dryer','laundry','LAUNDRY','Laundry'] ) and not(any(notname in s for notname in ['dish','DISH','Dish', 'Disw'] )) ]



trainingData["laundry"] = [any(laundry in feature for laundry in Washer) for feature in trainingData['features']]
labelEncoder = LabelEncoder()



trainingData['interest'] = labelEncoder.fit_transform(trainingData['interest_level'])



trainingDataSub = trainingData.loc[trainingData['interest']==0]



trainingDataSub = trainingDataSub.append(trainingData.loc[trainingData['interest']==1].sample(15000))



trainingDataSub = trainingDataSub.append(trainingData.loc[trainingData['interest']==2].sample(10000))





y = trainingData['interest']



X = trainingData[['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'numPics', 'has_ac', 'included_offer', 'concierge', 'laundry']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=52)
trainingData.dtypes
from sklearn import neural_network



#regr = neural_network.MLPClassifier(hidden_layer_sizes = (50,50,10))



#regr.fit(X_train, y_train)
#pred = regr.predict(X_test)



#print(confusion_matrix(pred, y_test))



#print(accuracy_score(pred, y_test))



#print(labelEncoder.classes_)

#regr.get_params(deep=True)
from sklearn.cluster import KMeans



regr = RandomForestClassifier(n_estimators=300, random_state=52)



regr.fit(X_train, y_train)



pred = regr.predict(X_test)



print(confusion_matrix(pred, y_test))



print(accuracy_score(pred, y_test))



print(labelEncoder.classes_)
testingData = pd.read_json('../input/test.json')



testingData['numPics'] = testingData['photos'].apply(len)
features = [x for sublist in testingData['features'] for x in sublist]



for x in features:

    if "*" in x: features.remove(x)



features = set(features)



features.discard('<null>')



has_ac = [ s for s in features if any(ac_name in s for ac_name in ['A/C', "AC", "Air Conditioning"] ) and not(any(wrong in s for wrong in ["FIRE",'ACT','APT', 'SPACE','YARD'])) ]



testingData['has_ac'] = [any(ac in feature for ac in has_ac) for feature in testingData['features'] ]

  

free_included = [ s for s in features if any(ac_name in s for ac_name in ["free","FREE","Free", "Gift", "gift", '1/2 Month fee', "included", "INCLUDED","Included"] ) ]

    

testingData["included_offer"] = [any(free in feature for free in free_included) for feature in testingData['features']]



doorman = [ s for s in features if any(ac_name in s for ac_name in ["doorman","DOORMAN","Doorman",'doormen','Doormen', 'full-service', 'concierge','Concierge','Attended Lobby', 'Attended lobby', 'attended lobby'] ) ]

   

testingData["concierge"] = [any(door in feature for door in doorman) for feature in testingData['features']]



Washer = [ s for s in features if any(ac_name in s for ac_name in ['Washer', "Dryer",'Washer','Dryer','washer','dryer','laundry','LAUNDRY','Laundry'] ) and not(any(notname in s for notname in ['dish','DISH','Dish', 'Disw'] )) ]



testingData["laundry"] = [any(laundry in feature for laundry in Washer) for feature in testingData['features']]
X = testingData[['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'numPics', 'has_ac', 'included_offer', 'concierge', 'laundry']]



predictions = regr.predict_proba(X)



output = pd.DataFrame(testingData['listing_id'], columns = ['listing_id'])



output['high'] = predictions[:,0]

output['low'] = predictions[:,1]

output['medium'] = predictions[:,2]





output.to_csv("submission1.csv", index=False)