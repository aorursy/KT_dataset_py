import numpy as np

import pandas as pd

from glob import glob

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt



from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

from sklearn.utils import shuffle 
import sys

sys.version
dataframe = glob("../input/activity-recognition/ActivityAccelerometer/*.csv")
#Dataset is loaded and merged, a new column named User_ID which gives the ID of the User after data is merged.



def load_data(dataframe):

    dataset = pd.DataFrame()

    for index,filename in enumerate(dataframe):

        df = pd.read_csv(filename, header=None)

        df['User_ID'] = index + 1

        dataset = dataset.append(df.iloc[:,1:])

    return dataset



data = load_data(dataframe)



#Names are given for the columns

data.columns = ['x_acceleration','y_acceleration','z_acceleration','Label','User_ID']



data.head()
#The minimum and the maximum values of each column is checked for Sanity Check

for i in data.columns:

    print("Maximum value of",i ,"is",data[i].max())

    print("Minimum value of",i ,"is",data[i].min())  
#User_id represents that there are 15 users

data['User_ID'].value_counts()
print(data.shape)

print(data.dtypes)
#No missing values

data.isnull().sum()
data.describe()
for i in data.columns:

    data[i].value_counts()
data.info()
data['Label']=data['Label'].replace(0, np.nan)

data.dropna(subset = ["Label"], inplace=True)

data['Label'] = data['Label'].astype(int)

print(data['Label'].value_counts())

print(data['Label'].unique())
plt.figure(figsize=(9,5))

sns.set(font_scale=1.2)

sns.heatmap(data.corr(), annot=True, annot_kws={"size":16})

plt.show()
plt.figure(figsize=[16,6])

sns.set_style('whitegrid') 

sns.distplot(data['x_acceleration'],kde=False, color ='blue') 

plt.title('Acceleration in the x-axis',fontsize=20)

plt.show()
plt.figure(figsize=[16,6])

sns.set_style('whitegrid') 

sns.distplot(data['y_acceleration'],kde=False, color ='red') 

plt.title('Acceleration in the y-axis',fontsize=20)

plt.show()
plt.figure(figsize=[16,6])

sns.set_style('whitegrid') 

sns.distplot(data['z_acceleration'],kde=False, color ='green') 

plt.title('Acceleration in the z-axis',fontsize=20)

plt.show()
plt.figure(figsize=[7,7])

Values_Label= data['Label'].value_counts()

plt.pie(Values_Label.values, labels=Values_Label.keys(), autopct='%0.2f')

plt.title('Label for each of the tasks',fontsize=20)

plt.show()
plt.figure(figsize=[15,5])

sns.set(style="darkgrid")

plt.title('User ID',fontsize=25)

ax = sns.countplot(x="User_ID", data=data)

plt.xlabel("User_ID", fontsize=20)

plt.ylabel("Count", fontsize=20)

plt.show()
data.groupby(['User_ID','Label']).size().unstack().plot(kind='bar',stacked=True, figsize=[15,7])

plt.title('Time spent on individual Tasks',fontsize=18)

plt.xlabel('User ID',fontsize=15)

plt.ylabel('Count of the Labels',fontsize=15)

plt.show()
Label_6 = data[(data['Label']==2) | (data['Label']==5 ) | (data['Label']==6)]
Label_6.groupby(['User_ID','Label']).size().unstack().plot(kind='bar', figsize=[15,7])

plt.title('Time spent on Tasks 2,5,6',fontsize=18)

plt.xlabel('User ID',fontsize=15)

plt.xticks(rotation=0)

plt.ylabel('Count of the Labels',fontsize=15)

plt.show()
sns.regplot(x=data["x_acceleration"], y=data["z_acceleration"], fit_reg=False)

plt.title('Relationship Acceleration in x-axis and z-axis',fontsize=18)

plt.xlabel('X acceleration',fontsize=15)

plt.ylabel('Z acceleration',fontsize=15)

plt.show()
sns.regplot(x=data["x_acceleration"], y=data["y_acceleration"], fit_reg=False)

plt.title('Relationship Acceleration in x-axis and y-axis',fontsize=18)

plt.xlabel('X acceleration',fontsize=15)

plt.ylabel('Y acceleration',fontsize=15)

plt.show()
sns.regplot(x=data["y_acceleration"], y=data["z_acceleration"], fit_reg=False)

plt.title('Relationship Acceleration in y-axis and z-axis',fontsize=18)

plt.xlabel('Y acceleration',fontsize=15)

plt.ylabel('Z acceleration',fontsize=15)

plt.show()
minimum_x_axis={}

maximum_x_axis={}



for i in data.Label.unique():

    maximum_x_axis[i]=(data[(data['Label']==i)]['x_acceleration'].max())

    minimum_x_axis[i]=(data[(data['Label']==i)]['x_acceleration'].min())





X = np.arange(len(minimum_x_axis))

fig, ax = plt.subplots(figsize=(12,6))

                     

ax.bar(X, minimum_x_axis.values(), width=0.4, color='#008B8B', align='center', label='minimum')

ax.bar(X-0.4, maximum_x_axis.values(), width=0.4, color='#48c9b0', align='center',label='maximum')



ax.legend()

plt.xticks(X, ['1','2','3','4','5','6','7'])

plt.xlabel("Labels",fontsize=15)

plt.ylabel("Value of acceleration",fontsize=15)

plt.title("Minimum and Maximum values of acceleration in X-axis for each Label", fontsize=17)

plt.show()

minimum_y_axis={}

maximum_y_axis={}



for i in data.Label.unique():

    maximum_y_axis[i]=(data[(data['Label']==i)]['y_acceleration'].max())

    minimum_y_axis[i]=(data[(data['Label']==i)]['y_acceleration'].min())





X = np.arange(len(minimum_y_axis))

fig, ax = plt.subplots(figsize=(12,6))

                     

ax.bar(X, minimum_y_axis.values(), width=0.4, color='#008B8B', align='center', label='minimum')

ax.bar(X-0.4, maximum_y_axis.values(), width=0.4, color='#48c9b0', align='center',label='maximum')



ax.legend()

plt.xticks(X, ['1','2','3','4','5','6','7'])

plt.xlabel("Labels",fontsize=15)

plt.ylabel("Value of acceleration",fontsize=15)

plt.title("Minimum and Maximum values of acceleration in Y-axis for each Label", fontsize=17)

plt.show()

 
minimum_z_axis={}

maximum_z_axis={}



for i in data.Label.unique():

    maximum_z_axis[i]=(data[(data['Label']==i)]['z_acceleration'].max())

    minimum_z_axis[i]=(data[(data['Label']==i)]['z_acceleration'].min())





X = np.arange(len(minimum_z_axis))

fig, ax = plt.subplots(figsize=(12,6))

                     

ax.bar(X, minimum_z_axis.values(), width=0.4, color='#008B8B', align='center', label='minimum')

ax.bar(X-0.4, maximum_z_axis.values(), width=0.4, color='#48c9b0', align='center',label='maximum')



ax.legend()

plt.xticks(X, ['1','2','3','4','5','6','7'])

plt.xlabel("Labels",fontsize=15)

plt.ylabel("Value of acceleration",fontsize=15)

plt.title("Minimum and Maximum values of acceleration in Z-axis for each Label", fontsize=17)

plt.show()

 
def User_plot(User):

    plt.figure()

    for i in range(User.shape[1]):

        plt.figure(figsize=(14,6))

        plt.subplot(User.shape[1],1,i+1)

        plt.plot(User[:,i],color='#008B8B')

        plt.show()

        
for i in data.User_ID.unique():

    print('User_ID',i)

    print('The acceleration in x,y,z axis and the Labels')

    User_plot(data[data.User_ID==i].iloc[:,:4].values)

    print('End of',i,'plot\n')
new_data=[]

for k,values in data.groupby('User_ID'):

    new_data.append(values.iloc[:,:4].values)



def activity_group(new_data,Labels):

    activity_groups=[{label:new[new[:,-1]==label] for label in Labels} for new in new_data]

    return activity_groups



def duration(activity_groups,Labels):

    frequency=52

    time_range = [[len(new[act])/frequency for new in activity_groups] for act in Labels]

    return time_range



def durations_plot(activity_groups,Labels):

    time_range = duration(activity_groups,Labels)

    plt.boxplot(time_range, labels=Labels)

    plt.title("Tasks grouped by their duration",fontsize=17 )

    plt.xlabel("Labels",fontsize=15)

    plt.ylabel("Duration of the Task",fontsize=15)

    plt.show()

    

Labels=[label for label in range(1,8)]

activity_groups=activity_group(new_data,Labels)

durations_plot(activity_groups,Labels)





X = data[['x_acceleration', 'y_acceleration', 'z_acceleration','User_ID']]

y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)



decision_tree_model = DecisionTreeClassifier(random_state=0)

clf = decision_tree_model.fit(X_train, y_train)

decision_pred = decision_tree_model.predict(X_test)



print("Accuracy score using Decision tree is",accuracy_score(y_test, decision_pred))
data_col = []

current_Mscore = 0.0

Num_columns = 4

shuf_columns = shuffle(range(0,Num_columns), random_state=1)

ls=['x_acceleration', 'y_acceleration', 'z_acceleration','User_ID']





for cols in range(0, Num_columns): 

    data_col.append(ls[shuf_columns[cols]])

    print(data_col)

    

    newData = data[data_col]

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(newData, data['Label'], test_size=0.30, random_state=0)

    dtree_classifier = DecisionTreeClassifier()



    fit = dtree_classifier.fit(X_Train, Y_Train)

    new_Score = dtree_classifier.score(X_Test, Y_Test)

    

    if new_Score < current_Mscore:

        data_col.remove(shuf_columns[cols])

    else:

        current_Mscore = new_Score

        print("Score with " + str(len(data_col)) + " selected features: " + str(new_Score))



print("There are " + str(len(data_col)) + " features selected:", data_col)
# Accuracy is put into a list to plot a graph for comparison

test_accuracy = decision_tree_model.score(X_test, y_test)

train_accuracy = decision_tree_model.score(X_train, y_train)



decision_score=[test_accuracy,train_accuracy]
cv = confusion_matrix(y_test, decision_pred)

print("Confusion matrix\n",cv)
decision_cr = classification_report(y_test,decision_pred)

print("Classification report\n",decision_cr)
target_names=[1,2,3,4,5,6,7]

#The values are normalised in order to plot the confusion matrix

decision_tree_cm = cv.astype('float') / cv.sum(axis=1)[:, np.newaxis]



#Confusion matrix is plotted for the normalised values obtained with the Labels

dtree_cm = pd.DataFrame(decision_tree_cm, columns=np.unique(target_names), index = np.unique(target_names))

dtree_cm.index.name = 'Actual values'

dtree_cm.columns.name = 'Predicted values'



plt.figure(figsize = (7,5))

sns.set(font_scale=1.4)

sns.heatmap(dtree_cm, cmap="Blues",linewidth=0.5, annot=True,annot_kws={"size":13})

plt.show()
non_optimal=[]



for optimal_k in range(1,20):

    KNN = KNeighborsClassifier(n_neighbors=optimal_k)

    KNN.fit(X_train,y_train)

    pred_optimal_k = KNN.predict(X_test)

    non_optimal.append(np.mean(pred_optimal_k != y_test))

    

plt.figure(figsize=(10,6))

plt.plot(range(1,20), non_optimal, '-ok')

plt.title('To find Optimal K value')

plt.xlabel('K')

plt.ylabel('Error')

plt.show()
KNN_classifier = KNeighborsClassifier(8, weights='distance')

KNN_classifier.fit(X_train, y_train)

y_pred = KNN_classifier.predict(X_test)



print("Accuracy score using KNN Classifier is: {}".format(accuracy_score(y_test, y_pred)))
data_cols = []

current_MScore = 0.0

number_cols = 4

shuffle_data_cols = shuffle(range(0,number_cols), random_state=1)

ls = ['x_acceleration', 'y_acceleration', 'z_acceleration','User_ID']





for cols in range(0, number_cols): 

    data_cols.append(ls[shuffle_data_cols[cols]])

    newData = data[data_cols]

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(newData, data['Label'], test_size=0.30, random_state=0)

    

    KNN_classifier = KNeighborsClassifier(8, weights='distance')

    fit = KNN_classifier.fit(X_TRAIN, Y_TRAIN)

    present_Score = KNN_classifier.score(X_TEST, Y_TEST)

    

    if present_Score < current_MScore:

        data_cols.remove(shuffle_data_cols[cols])

    else:

        current_MScore = present_Score

        print("Score with " + str(len(data_cols)) + " selected features: " + str(present_Score))



print("There are " + str(len(data_cols)) + " features selected:", data_cols)
KNN_classifier = KNeighborsClassifier(8, weights='distance', p=1)

KNN_classifier.fit(X_train, y_train)

y_pred = KNN_classifier.predict(X_test)



print("Accuracy score using KNN Classifier after parameter tuning is",accuracy_score(y_test, y_pred))
KNN_classifier_param = KNeighborsClassifier(8, weights='distance', p=1)

KNN_classifier_param.fit(X_train, y_train)

y_pred = KNN_classifier_param.predict(X_test)



print("Accuracy score using KNN Classifier is: {}".format(accuracy_score(y_test, y_pred)))
test_accuracy = KNN_classifier_param.score(X_test, y_test)

train_accuracy = KNN_classifier_param.score(X_train, y_train)



KNN_score=[test_accuracy,train_accuracy]
k_confusion_matrix = confusion_matrix(y_test, y_pred)

print("Confusion matrix is\n",k_confusion_matrix)
target_names=[1,2,3,4,5,6,7]

#The values are normalised in order to plot the confusion matrix

conf_matrix = k_confusion_matrix.astype('float') / k_confusion_matrix.sum(axis=1)[:, np.newaxis]



#Confusion matrix is plotted for the normalised values obtained with the Labels

df_cm_k = pd.DataFrame(conf_matrix, columns=np.unique(target_names), index = np.unique(target_names))

df_cm_k.index.name = 'Actual values'

df_cm_k.columns.name = 'Predicted values'



plt.figure(figsize = (7,5))

sns.set(font_scale=1.4)

sns.heatmap(df_cm_k, cmap="Greens", annot=True,annot_kws={"size": 13})

plt.show()
knn_cr=classification_report(y_test, y_pred)

print("Classification report\n",knn_cr)
ind = np.arange(2) 

width = 0.35

plt.figure(figsize=(10,6))

plt.bar(ind, decision_score, width, label='decision_score')

plt.bar(ind + width, KNN_score, width, label='KNN_score')



plt.ylabel('Accuracy')

plt.title('Comparison of Accuracy ')



plt.xticks(ind + width / 2, ('Test_data', 'Train_data'))

plt.legend(loc='best')

plt.show()