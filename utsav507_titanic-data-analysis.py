import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 
import os 
import gc 
import matplotlib.pyplot as plt 
import seaborn as sns
import math
import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
# Creating the INPUT FOLDER which will contain our input files!
INPUT_FOLDER='/Users/Utsav/Projects/Udacity DAND/titanic-data-analysis/'
print ('File Sizes:')
for f in os.listdir(INPUT_FOLDER):
   if 'zip' not in f:
      print (f.ljust(30) + str(round(os.path.getsize(INPUT_FOLDER +  f) / 1000, 2)) + ' KB')
main_file = pd.read_csv(INPUT_FOLDER + 'titanic_train.csv')
# Head of the dataset
main_file.head()
# Description of the main_file
main_file.describe()
# Check the 
main_file.isnull().sum()
# Find length of NaN values in Age
len(np.argwhere(np.isnan(main_file["Age"])))
# Find length of the column "Age"
len(main_file["Age"])
# View the NaN in Age for Female-Sex
female_age = main_file[main_file["Sex"] == "female"]["Age"]
female_age.tail()
# View the NaN in Age for Male-Sex
male_age = main_file[main_file["Sex"] == "male"]["Age"] 
male_age.head()
# Store median value into variable
female_median = female_age.median()
male_median = male_age.median()
# Assign age and sex into lists
age = list(main_file["Age"])
sex = list(main_file["Sex"])
# Run loop to replace the NaN values with the median value of the column
# The NaN values in female ages are replaced by the median of the female ages
# The NaN values in male ages are replaced by the median of the male ages
     
# Get the median by sex
medians = main_file.groupby('Sex')['Age'].median()
# Set dataframe index by sex
main_file = main_file.set_index(['Sex'])
# Fill na
main_file['Age'] = main_file['Age'].fillna(medians)
# if you want to reset the index
main_file = main_file.reset_index()        
# Create Survival Label Column
# main_file['Survival'] = main_file.Survived.map({0 : 'Died', 1 : 'Survived'})
# main_file.Survival.head()
# Create Pclass Label Column
# main_file['Class'] = main_file.Pclass.map({1 : 'First Class', 2 : 'Second Class', 3 : 'Third Class'})
# main_file.Class.head()
# Create Embarked Labels Column
# main_file['Ports'] = main_file.Embarked.map({'C' : 'Cherbourg', 'Q' : 'Queenstown', 'S' : 'Southampton'})
# main_file.Ports.head()
# Total number of passengers
no_of_passengers = main_file.groupby("Survived").size()
no_of_passengers
# Total number of survivors
no_of_survivors = no_of_passengers[1]
no_of_survivors
# Total number of deaths
no_of_deaths = no_of_passengers[0]
no_of_deaths
# Percentage of total survivors and deaths
percent_of_survivors = (no_of_survivors/(no_of_survivors + no_of_deaths))*100
percent_of_deaths = (no_of_deaths/(no_of_survivors + no_of_deaths))*100
percent_of_survivors
percent_of_deaths
# Plot bar graph for Total Number of Passengers - Deaths and Survived

objects = ('Number of Deaths', 'Number of Survivors')
bars = np.arange(len(objects))
no_of_passengers.plot(kind = "bar", figsize = (10,8), color = 'r')
plt.xticks(bars, objects, rotation='horizontal')
plt.ylabel("Number of Passengers")
plt.xlabel("Total Number of Passengers")
plt.title("Total Number of Passengers - Deaths and Survived on Titanic")
plt.show()
# The following function is used to create counts and percentages in the pie
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct
# Plot pie chart for Total Number of Passenger Deaths and Survived

df_survived = main_file.groupby('Survived').size()

colors = ['gold', 'yellowgreen', 'lightskyblue', 'lightcoral']
plt.pie(df_survived, shadow = True, colors = colors, labels = ['Died', 'Survived'], autopct = make_autopct(df_survived))
plt.title('Total Number of Passengers - Deaths and Survived on Titanic')
# Grouping by Sex
male_female = main_file.groupby("Sex").size()
male_female
# View number of females and males deaths and survived
survived_male_female = main_file.groupby(("Sex","Survived")).size()
survived_male_female
# Plot pie chart for Total Number of Females and Males - Deaths and Survived

df_survived = main_file.groupby(('Survived', 'Sex')).size()

colors = ['gold', 'yellowgreen', 'lightskyblue', 'lightcoral']
plt.pie(df_survived, shadow = True, colors = colors, labels = ['Number of Female - Deaths', 'Number of Female - Survivors', 'Number of Male - Deaths', 'Number of Male - Survivors'], autopct = make_autopct(df_survived))
plt.title('Total Number of Females and Males - Deaths and Survived on Titanic')
# Plot bar graph for Total Number of Females and Males - Deaths and Survived
objects = ('Number of Female - Deaths', 'Number of Female - Survivors', 'Number of Male - Deaths', 'Number of Male - Survivors')
bars = np.arange(len(objects))
survived_male_female.plot(kind = "bar", figsize = (10,8), color = 'g')
plt.xticks(bars, objects, rotation='horizontal')
plt.ylabel("Number of Passengers")
plt.xlabel("Total Number of Passengers")
plt.title("Total Number of Females and Males - Deaths and Survived on Titanic")
plt.show()
print ("Percent of Female Passengers Survived:\t{}".format((survived_male_female["female"][1]/male_female["female"])*100))
print ("Percent of Male Passengers Survived:\t{}".format((survived_male_female["male"][1]/male_female["male"])*100))
age_of_passengers = main_file.groupby("Age").size()
age_of_passengers.head()
# Plot bar graph for Age of passengers on titanic

age_of_passengers.plot(kind = "bar", figsize = (20,8))
plt.ylabel("Number of Passengers")
plt.xlabel("Age")
plt.title("Total Number of Passengers")
plt.show()
main_file["Age"].describe()
# Group by Sibling/Spouse Traveling with their Sibling/Spouse
sibsp = main_file.groupby("SibSp").size()
# Plot bar graph for Number of Passengers Traveling with their Sibling/Spouse

sibsp.plot(kind = "bar", figsize = (10,8), color = 'c')
plt.ylabel("Number of Passengers")
plt.xlabel("Number of Siblings/Spouse")
plt.title("Number of Passengers Traveling with their Sibling/Spouse")
plt.show()
# Group data by Passengers traveling with their Parents/Children
parch = main_file.groupby("Parch").size()
parch
# Bar plot of Number of Passengers traveling with their Parents/Children

parch.plot(kind = "bar", figsize = (10,8), color = 'r')
plt.ylabel("Number of Passengers")
plt.xlabel("Number of Parents/Children")
plt.title("Number of Passengers traveling with their Parents/Children")
plt.show()
# Group data by Number of Passengers Traveling in Different Socio-Economic Classes (Pclass)
pclass = main_file.groupby("Pclass").size()
pclass
# Bar plot of Number of Passengers Traveling in Different Socio-Economic Classes

pclass.plot(kind = "bar", figsize = (10,8), color = 'y')
plt.ylabel("Number of Passengers")
plt.xlabel("Pclass")
plt.title("Number of Passengers Traveling in Different Socio-Economic Classes")
plt.show()
# Group data by Number of Passengers Traveling in Class 1, 2, 3 - Deaths and Survivors

class_survived = main_file.groupby(["Pclass","Survived"])
class_survived
# Bar plot of Number of Passengers Traveling in Class 1, 2, 3 - Deaths and Survivors

objects = ('Pclass 1', 'Pclass 2', 'Pclass 3')
bars = np.arange(len(objects))
class_survived.size().unstack().plot(kind = "bar", figsize = (10,8))
plt.xticks(bars, objects, rotation='horizontal')
plt.ylabel("Number of Passengers")
plt.xlabel("Number of Passengers in Pclass")
plt.title("Number of Passengers Traveling in Class 1, 2, 3 - Deaths and Survivors")
plt.legend(['Died', 'Survived'])
plt.show()
# def bar_chart_per_class(class_titanic):
#     """
#     Produces a bar chart of dead and surviving passengers in the class passed as argument.
#     """
# 
#     objects = ('Number of Deaths', 'Number of Survivors')
#     bars = np.arange(len(objects))
#     class_survived[class_titanic].plot(kind = "bar", figsize = (10,8), color = "m")
#     plt.xticks(bars, objects, rotation='horizontal')
#     plt.ylabel("Number of Passengers")
#     plt.xlabel("Number of Passengers in Class {}".format(class_titanic))
#     plt.title("Number of Passengers Traveling in Class {} - Deaths and Survivors".format(class_titanic))
#     plt.show()
# 
# bar_chart_per_class(1)
# bar_chart_per_class(2)
# bar_chart_per_class(3)
# Box Plot of Survivors from Pclass v/s Fare

ax = sns.boxplot(x = "Pclass", y = "Fare", hue = "Survived", data = main_file, palette = "Set1", width = 0.7, fliersize = 3, whis = 1.5, linewidth = 0.5) 
plt.title("Box Plot of Survivors from Pclass v/s Fare")
plt.legend(["Dead (Red)", "Survived (Blue)"])
sns.plt.show()
# Group data by Embarkment 

embark_survived = main_file.groupby(("Embarked", "Survived")).size()
embark_survived
# Bar plot of Distribution of Survivors with respect to Embarkment

embark_survived.plot(kind = "bar", figsize = (10,8), color = "r")
plt.ylabel("Number of Passengers")
plt.xlabel("Embarked, Survived")
plt.title("Distribution of Survivors with respect to Embarkment")
plt.show()
# Group data by Survived and Fare

fare_survived = main_file.groupby(("Survived", "Fare")).size()
fare_survived.head()
# Box plot of Number of Passengers that Died and Survived with respect to the Fare

ax = sns.boxplot(x = "Survived", y = "Fare",data = main_file, palette = "Set1", width = 0.7, fliersize = 3, whis = 1.5, linewidth = 0.5) 
plt.title("Number of Passengers that Died and Survived with respect to the Fare")
plt.legend(["Dead (Red)","Survived (Blue)"])
sns.plt.show()
# Group data by Cabin
cabin = main_file.groupby("Cabin").size()
# Creating new Cabin_Deck by extracting first letter from Cabin column of original dataset

main_file["Cabin_Deck"] = main_file["Cabin"].astype(str).str[0]
cabin_deck = main_file.groupby("Cabin_Deck").size()
cabin_deck
cabin_deck_survived = main_file.groupby(("Cabin_Deck", "Survived")).size()
# Bar plot Distribution of Deck (Cabin) and Survived

cabin_deck_survived.drop("n").unstack().plot(kind = "bar", figsize = (10,8))
plt.ylabel("Number of Passengers")
plt.xlabel("Cabin Deck")
plt.title("Distribution of Survivors and Deaths with respect to Deck")
plt.legend(['Died', 'Survived'])
plt.show()
# Group data by Pclass and Cabin

pclass_cabin = main_file.groupby(("Pclass", "Cabin_Deck")).size()
# Describe how the grouped data looks
pclass_cabin.describe()
# Bar plot Distribution of Pclass and Cabin Passengers traveling on titanic

pclass_cabin.unstack().plot(kind = "bar", figsize = (100,8), color = 'g')
plt.ylabel("Number of Passengers")
plt.xlabel("Pclass, Cabin")
plt.title("Distribution of Pclass and Cabin Deck")
plt.show()
# Only keep title from Name column

Title = []

for i in main_file["Name"]:
    Title.append(i.split(",")[1].split(".")[0])
main_file["Name"] = Title
main_file["Name"].head()
# Delete columns that will not be necessary for prediction algorithms
del main_file["PassengerId"]
del main_file["Ticket"]
del main_file["Cabin"]
main_file.head()
# Convert string values to int values for the ease of prediction
main_file = pd.get_dummies(main_file)
main_file.head()
# View data - ready for training and testing
pd.isnull(main_file).sum()
# Split data into 80 - 20 
X = main_file.iloc[:,1:]
Y = main_file["Survived"]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2)
# Check length of train and test data
len(X_train)
len(X_test)
len(Y_train)
len(Y_test)
# Random Forest Classification Algorithm applied to train data
RF = RandomForestClassifier(n_jobs = 2)
RF.fit(X_train, Y_train)
RF.score(X_train, Y_train)
# Predict Random Forest Algorithm on Test Data
predictions_RF = RF.predict(X_test)
# Print Accuracy Score for Random Forest Algorithm
print("Accuracy Score is: ")
print(accuracy_score(Y_test, predictions_RF))
print()
# Classification Report of Prediction
print("Classification Report: ")
print(classification_report(Y_test, predictions_RF))
# Confusion Matrix for predictions made
conf = confusion_matrix(Y_test, predictions_RF)
conf
# Plot Confusion Matrix for Random Forest Classification Algorithm
label = ["0", "1"]
sns.heatmap(conf, annot = True, xticklabels = label, yticklabels = label)
plt.xlabel("Prediction - Random Forest Classification Algorithm")
plt.title("Confusion Matrix for Random Forest Classification Algorithm")
plt.show()
# LDA applied to train data
lda = LinearDiscriminantAnalysis()
lda.fit(X_train,Y_train)
predictions_lda = lda.predict(X_test)
# Print Accuracy Score for LDA
print("Accuracy Score is:")
print(accuracy_score(Y_test, predictions_lda))
print()
# Classification Report of Prediction
print("Classification Report:")
print(classification_report(Y_test, predictions_lda))
# Confusion Matrix for predictions made
conf2 = confusion_matrix(Y_test,predictions_lda)
conf2
# Plot Confusion Matrix for Linear Discrimination Analysis
label = ["0","1"]
sns.heatmap(conf2, annot=True, xticklabels=label, yticklabels=label)
plt.xlabel("Prediction - Linear Discrimination Analysis")
plt.title("Confusion Matrix for Linear Discrimination Analysis")
plt.show()