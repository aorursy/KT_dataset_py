import pandas as pd 

import numpy as np 

from pandas import DataFrame

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

data=pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

data.drop(labels='reservation_status_date',axis=1,inplace=True)



Y=np.array(data['is_canceled'])



data=pd.get_dummies(data,prefix_sep='_',drop_first=True)



X=data.drop(labels=['is_canceled','company','agent','children'],axis=1,inplace=False)

#data=data.fillna(data.mean(),inplace=True)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.1, random_state=5)



from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=100,max_features=15)



from sklearn.model_selection import cross_val_score

resul=cross_val_score(rfc,X_train,Y_train,scoring='accuracy')

rfc.fit(X_train,Y_train)

predictions=rfc.predict(X_test)

score=accuracy_score(predictions,Y_test)

print("Accuracy: " ,(score*100))