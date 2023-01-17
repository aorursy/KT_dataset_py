# Displying image



from IPython.display import Image

Image("../input/dtdata/DT.PNG")
# Importing libraries



import pandas as pd
# Reading file



print(r"Reading 'daily_weather.csv' file...")

df=pd.read_csv("../input/dtdata/daily_weather.csv")

print("Reading few records from the dataset")

df.head()
# Informatin about dataset



print("Informatin about dataset.")

df.info()
# Listing all the records having NaN



print("Listing all the records having NaN")

df[df.isnull().any(axis=1)]
# Dropping the Column 'Number' as it is not useful



df.drop('number',axis=1,inplace=True)

df.info()

print(r"The column 'number' is removed but the count of records of each column is different due to presece of NaN")
# Dropping all the records having NaN



print("Removing all the records having NaN")

df.dropna(inplace=True)



df.info()

print("Now the count of all the records in each column is equal i.e. 1064")
# Creating copy of the original dataset for processing



print("Creating copy of the original dataset for processing")

df2=df.copy()



print("Assigning the records haviing relative_humidity_3pm > 25")

df2['humidityval']=(df2['relative_humidity_3pm']>25)*1



df2.head()
# Dropping the column 'relative_humidity_3pm' form the df2



print(r"Dropping the column 'relative_humidity_3pm' form the df2")

df2.drop('relative_humidity_3pm',axis=1, inplace=True)

df2.head()
# Assigning dataframs to variable



print("Assigning dataframs to variable.")

y=df2['humidityval']

x=df2.drop('humidityval', axis=1)

df2.head()
# Importing libraries



print("Importing libraries")

from sklearn.model_selection import train_test_split



print("Spliting the dataset")

trainx, testx, trainy, testy=train_test_split(x,y,test_size=0.33)

print("Spliting the dataset completed")
# Importing libraries for DecisionTreeClassifier



print("Importing libraries for DecisionTreeClassifier")

from sklearn.tree import DecisionTreeClassifier



print("Creating instace of the model")

dtcModel=DecisionTreeClassifier()



print("Training the model")

dtcModel.fit(trainx,trainy)

print("Model Trained")



# Predicting the values



print("Predicting the values..")

predictedOp=dtcModel.predict(testx)

print("Predicting the values completd.")



# Checking the accuracy



print("Checking the accuracy")

from sklearn.metrics import accuracy_score

print("Accuracy is: %s" % accuracy_score(testy,predictedOp))



# Printing confusion matrix



from sklearn.metrics import confusion_matrix

print("Confusion Matrix: \n %s" % confusion_matrix(testy,predictedOp))
# Importing libraries for RandomForestClassifier



print("Importing libraries for RandomForestClassifier")

from sklearn.ensemble import RandomForestClassifier



print("Creating instace of the model")

rfcModel=RandomForestClassifier(n_estimators=30)



print("Training the model")

rfcModel.fit(trainx,trainy)

print("Model Trained")



# Predicting the values



print("Predicting the values..")

predictedOpRf=rfcModel.predict(testx)

print("Predicting the values completd.")



# Checking the accuracy



print("Checking the accuracy")

from sklearn.metrics import accuracy_score

print("Accuracy is: %s" % accuracy_score(testy,predictedOpRf))



# Printing confusion matrix



from sklearn.metrics import confusion_matrix

print("Confusion Matrix: \n %s" % confusion_matrix(testy,predictedOpRf))
# Difference in prediction value between Decision Tree and Random Forest algorithm



print ("Difference in prediction value between Decision Tree and Random Forest algorithm")

print("Accuracy is for Decision Tree: %s" % accuracy_score(testy,predictedOp))

print("Accuracy is for Random Forest: %s" % accuracy_score(testy,predictedOpRf))

print("Confusion Matrix Decision Tree: \n %s" % confusion_matrix(testy,predictedOp))

print("Confusion Matrix Random Forest: \n %s" % confusion_matrix(testy,predictedOpRf))
print("Notebook completd!")