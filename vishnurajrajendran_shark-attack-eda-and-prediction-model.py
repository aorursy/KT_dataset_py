#Immport neccessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Import sklearn machine learning packages
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
shark_attack = pd.read_csv("../input/sharkattack.csv")
shark_attack.drop(["Unnamed: 0"],inplace = True,axis=1)
shark_attack = shark_attack[shark_attack['Fatal'] != "UNKNOWN"]
shark_attack.describe()
shark_attack['Date_formatted'] = pd.to_datetime(shark_attack['Date'])
shark_attack["Fatal_val"] = shark_attack["Fatal"].apply(lambda Fatal: 0 if Fatal == 'N' else 1)
shark_attack["Hemisphere_val"] = shark_attack["Hemisphere"].apply(lambda Hemisphere: 0 if Hemisphere == 'N' else 1)
shark_attack.drop(['Date','Country code','Fatal'],inplace=True,axis=1)
shark_attack['year'] = shark_attack['Date_formatted'].dt.year
shark_attack['month'] = shark_attack['Date_formatted'].dt.month
shark_attack['day'] = shark_attack['Date_formatted'].dt.day
shark_attack['week'] = shark_attack['Date_formatted'].dt.week
shark_attack['day_of_week'] = shark_attack['Date_formatted'].dt.dayofweek
dayOfWeek = {0 : 'Monday',1 : 'Tuesday' ,2: 'Wednesday' ,3 : 'Thursday' ,4 : 'Friday',5 : 'Saturday' ,6 : 'Sunday'}
shark_attack['weekday'] = shark_attack['Date_formatted'].dt.dayofweek.map(dayOfWeek)
monthOfYear = {1: 'Jan',2: 'Feb',3: 'Mar',4: 'Apr',5: 'May',6: 'Jun',7: 'Jul',8: 'Aug',9: 'Sep',10: 'Oct',11: 'Nov',12: 'Dec'}
shark_attack['Month'] = shark_attack['Date_formatted'].dt.month.map(monthOfYear)
shark_attack.set_index(['Date_formatted'],inplace = True)
#Plot Total Fatal and Non-fatal incident count
plt.figure(figsize=(8,6))
ax = sns.countplot(x="Fatal_val", data=shark_attack)
plt.xlabel('Fatal')
plt.ylabel('No of Incidents')
plt.title('Fatal vs Count')
plt.show()
#Plot Total Fatal and Non-fatal incident count
plt.figure(figsize=(8,6))
sns.countplot(x="weekday", hue="Fatal_val", data=shark_attack);
plt.xlabel('Day of Week')
plt.ylabel('No of Incidents')
plt.title('Weekday vs Count')
plt.show()
plt.figure(figsize=(8,6))
sns.countplot(x="Hemisphere", hue="Fatal_val", data=shark_attack);
plt.xlabel('Hemisphere')
plt.ylabel('No of Incidents')
plt.title('Hemisphere vs Count')
plt.show()
plt.figure(figsize=(10,8))
sns.countplot(x="Month", hue="Fatal_val", data=shark_attack);
plt.xlabel('Month')
plt.ylabel('No of Incidents')
plt.title('Month vs Count')
plt.show()
shark_attack.dropna(axis = 0, inplace = True)
shark_attack.describe()
shark_attack['Type'] = shark_attack['Type'].astype('category')
shark_attack['Activity'] = shark_attack['Activity'].astype('category')
cat_columns = shark_attack.select_dtypes(['category']).columns
shark_attack[cat_columns] = shark_attack[cat_columns].apply(lambda x: x.cat.codes)
shark_attack.reset_index(inplace=True)
shark_attack.columns
X = shark_attack[['Type','Activity', 'Hemisphere_val', 'year', 'month', 'day','week', 'day_of_week']]
y = shark_attack.Fatal_val
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=1)
dt=DecisionTreeClassifier()
dt_model=dt.fit(X_train,y_train)
dt_prd=dt_model.predict(X_test)
print(accuracy_score(y_test,dt_prd))
rf_clf=RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 10, random_state = 123)
rf_model=rf_clf.fit(X_train,y_train)
rf_prediction=rf_model.predict(X_test)
print(accuracy_score(y_test,rf_prediction))
#confusion_matrix(y_test,rf_prediction)
rf_confusion = metrics.confusion_matrix(y_test,rf_prediction)
dt_confusion = metrics.confusion_matrix(y_test,dt_prd)

print("Random Forest Confusion Matrix :")
print(rf_confusion)
print("------------------------------------")
print("Decision Tree Confusion Matrix :")
print(dt_confusion)
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

print('Random Forest Confusion Matrix')
plot_confusion_matrix(rf_confusion)
print('Decision Tree Confusion Matrix')
plot_confusion_matrix(dt_confusion)
