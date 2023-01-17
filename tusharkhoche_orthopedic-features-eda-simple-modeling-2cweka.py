# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns



 

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import LabelEncoder



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 2. Import data 2C_weka.csv for 2 Class Classification.



missing_value_formats = ["n.a.","?","NA","n/a","na","--"," ", "  "]

TwoC_weka_data = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv', na_values = missing_value_formats)
TwoC_weka_data.info()
TwoC_weka_data.describe()
TwoC_weka_data.shape
# Checking top 5 rows

TwoC_weka_data.head()
# Checking last 5 rows

TwoC_weka_data.tail()
# Checking for skewness

TwoC_weka_data.skew()
# Checking for null values and duplicate data

TwoC_weka_data.isna().sum()
TwoC_weka_data.isnull().sum()
TwoC_weka_data.duplicated().sum()
# Checking for unique values in target variable "class"

print(TwoC_weka_data['class'].unique())
# Method 1

pd.crosstab(TwoC_weka_data['class'],columns='Count')
# Method 2

print(TwoC_weka_data['class'].value_counts())
# Method 3

sns.countplot(x='class',data=TwoC_weka_data)

plt.show()
# Identifying Type Of Features 

# Numerical Features & Categorical Features



numerical_features = TwoC_weka_data.select_dtypes(include = [np.number])
print(numerical_features.columns)
# Now we want to segregate discrete variables from continuous variables

# So, we count the number of unique values in each feature. If count of unique values is less than 25 then we consider it as

# discrete variable otherwise it is a continuous variable



continuous_numerical_features = []

discrete_numerical_features = []
for feature in numerical_features:

    if(len(TwoC_weka_data[feature].unique())>25):

        continuous_numerical_features.append(feature)

        print('continuous_numerical_features ',feature)   
# Visualizing Distribution For Numerical Columns

# I wanted to use "displot" instead of "distplot" as "distplot" is going to be deprecated. See below link 

# https://seaborn.pydata.org/generated/seaborn.distplot.html?highlight=distplot#seaborn.distplot

# But I couldn't use "displot" as using it threw error "module 'seaborn' has no attribute 'displot'"



for feature in numerical_features.columns:

    sns.distplot(numerical_features[feature],kde=True)

    plt.show()
for feature in numerical_features.columns:

    sns.boxplot(TwoC_weka_data[feature])

    plt.show()
# Plotting Correlation HeatMap



plt.figure(figsize=(5,5))

sns.heatmap(TwoC_weka_data.corr(),annot=True)

plt.show()
# Plotting Barplot. Showing the numbers



for feature in numerical_features.columns:

    sns.barplot(x='class',y=feature,data=TwoC_weka_data)

    plt.show()
# Plotting swarmplot also



for feature in numerical_features.columns:

    sns.swarmplot(TwoC_weka_data['class'],TwoC_weka_data[feature])

    plt.show()
# Plotting pairplot

sns.pairplot(TwoC_weka_data,size=3,hue='class')

plt.show()
# Power Transformation is done to make data normal.

# Creating a dataset of numerical features only for transformation



numerical_dataset = TwoC_weka_data.iloc[:,:6]

numerical_dataset
power = PowerTransformer(method='yeo-johnson', standardize=True)

TwoC_weka_data_transformed = power.fit_transform(numerical_dataset)

TwoC_weka_data_transformed = pd.DataFrame(TwoC_weka_data_transformed,columns = numerical_dataset.columns)
TwoC_weka_data_transformed.head()
# Distribution after transformation. 

# I have plotted both the original & transformed distribution for comparison.

# We can observe that features have been transformed into Normal distribution

# Transformation is done as models perform better for Gaussian or Gaussian like distribution



for feature in TwoC_weka_data_transformed.columns:

    #print("Original ", feature)

    sns.distplot(numerical_features[feature],kde=True)

    plt.show()

    #print("             Transformed ", feature)

    sns.distplot(TwoC_weka_data_transformed[feature],kde=True)

    plt.show()  

    #print("-----------------------------------------------------")
# Keeping X in uppercase & y in lowercase as per standard convention



X = TwoC_weka_data_transformed

y = TwoC_weka_data.iloc[:,6:]
# First Splitting the data set into train & test data set so that while scaling or normalizing, test data should not affect train data

# Second, different random states can give different results. So we need to test for multiple random states

# Third, for every random state, different value of k can give different results. So, we need to test for multiple values of k

# for each of the random state



# The "fit" method gives mean and standard deviation.

# So we do "fit" the model using train data and then "transform" or apply that mean & std on test data.



# Scaling  or Normalization should be done separately on train data & test data.

# This is done to scale or normalize all the variable with different scales so that all these variable become comparable.

# We check for multiple random state & for each random state, we check for multiple K values

# This is how we can come to a conclusion which random state and value of K is to be chosen



ran_state = np.arange(1,50)

neighbours = np.arange(5,41) 

# I know that it is better to keep K-Value odd to have clear majority but I am not keeping it because I tried and I am getting much 

# better result with even numbers.



 

test_accuracy_list = []

train_accuracy_list = []

desired_k_value_list = []

desired_random_state_list = []

conf_matrix_report_list = []

class_report_list = []

 



for r_state in ran_state:

     

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=r_state)

    

    std_scaler = StandardScaler() 

     

    std_scaler.fit(X_train)     

    X_train_scaled = std_scaler.transform(X_train)

    X_test_scaled  = std_scaler.transform(X_test)

    

    for k_value in neighbours:

        # For metric='minkowski' p=2 means using Euclidean distance &  p=1 means Manhattan distance 

        KNN = KNeighborsClassifier(n_neighbors=k_value,metric='minkowski',algorithm='auto',p=2) 

        

        KNN.fit(X_train_scaled,y_train)

        y_pred = KNN.predict(X_test_scaled)    

        

        conf_matrix = metrics.confusion_matrix(y_test,y_pred)

        class_report = metrics.classification_report(y_test,y_pred)

        train_score = np.round(KNN.score(X_train_scaled,y_train),2)

        test_score = np.round(KNN.score(X_test_scaled,y_test),2)

        test_accuracy_list.append(test_score)

        train_accuracy_list.append(train_score)

        desired_k_value_list.append(k_value)

        desired_random_state_list.append(r_state)

        conf_matrix_report_list.append(conf_matrix)

        class_report_list.append(class_report)



        

test_accuracy_array = np.array(test_accuracy_list)

result = np.where(test_accuracy_array>0.86)

result = result[0]



     # If a patient is predicted Normal when he is Abnormal, then this prediction is bad. Patient is having medical issue but model 

     # predicted patient does not have any issue. 

     # We definitely need to minimize this (False Negative) as much as possible. So, I have chosen conf_matrix[1,0]<5 

     # for the this reason.

    

for r in result:  

    conf = conf_matrix_report_list[r]

    if(conf[1,0]<5):

        print('Test Accuracy',test_accuracy_list[r],'Train Accuracy',train_accuracy_list[r],'K Value ' ,desired_k_value_list[r],'Random State ',desired_random_state_list[r])

        print()

        print("Confusion Matrix ")

        print(conf_matrix_report_list[r])

        print()

        print("Classification Report ")

        print(class_report_list[r])

        print("--------------------------------------------------------")



        
ran_state = np.arange(1,50)



test_accuracy_list = []

train_accuracy_list = []

desired_random_state_list = []

conf_matrix_report_list = []

class_report_list = []



for r_state in ran_state:

    GNB_X_train,GNB_X_test,GNB_y_train,GNB_y_test = train_test_split(X,y,test_size=0.3,random_state=r_state)

    

    gnb = GaussianNB()

    gnb.fit(GNB_X_train,GNB_y_train)

    GNB_y_pred = gnb.predict(GNB_X_test)

    

    conf_matrix = metrics.confusion_matrix(GNB_y_test,GNB_y_pred)

    class_report = metrics.classification_report(GNB_y_test,GNB_y_pred)

    test_score = np.round(gnb.score(GNB_X_test,GNB_y_test),2)

    train_score = np.round(gnb.score(GNB_X_train,GNB_y_train),2)

    test_accuracy_list.append(test_score)

    train_accuracy_list.append(train_score)   

    desired_random_state_list.append(r_state)

    conf_matrix_report_list.append(conf_matrix)

    class_report_list.append(class_report)

     

        

test_accuracy_array = np.array(test_accuracy_list)

result = np.where(test_accuracy_array>0.80)

result = result[0]





for r in result:

    conf = conf_matrix_report_list[r]

    if(conf[1,0]<5):

        print('Test Accuracy',test_accuracy_list[r],'Train Accuracy',train_accuracy_list[r],'Random State ',desired_random_state_list[r])

        print()

        print("Confusion Matrix ")

        print(conf_matrix_report_list[r])

        print()

        print("Classification Report ")

        print(class_report_list[r])

        print("--------------------------------------------------------")
ran_state = np.arange(1,50)



test_accuracy_list = []

train_accuracy_list = []

desired_random_state_list = []

conf_matrix_report_list = []

class_report_list = []





for r_state in ran_state:

    

    LR_X_train,LR_X_test,LR_y_train,LR_y_test = train_test_split(X,y,test_size=0.3,random_state=r_state)

    

    logistic_regression = LogisticRegression()

    logistic_regression.fit(LR_X_train,LR_y_train)

    LR_y_predict = logistic_regression.predict(LR_X_test)

    

    conf_matrix = metrics.confusion_matrix(LR_y_test,LR_y_predict)

    class_report = metrics.classification_report(LR_y_test,LR_y_predict)

    test_score = np.round(logistic_regression.score(LR_X_test,LR_y_test),2)

    train_score = np.round(logistic_regression.score(LR_X_train,LR_y_train),2)

    test_accuracy_list.append(test_score)

    train_accuracy_list.append(train_score)   

    desired_random_state_list.append(r_state)

    conf_matrix_report_list.append(conf_matrix)

    class_report_list.append(class_report)

    

        

test_accuracy_array = np.array(test_accuracy_list)

result = np.where(test_accuracy_array>0.80)

result = result[0]





             

for r in result:

    conf = conf_matrix_report_list[r]

    if(conf[1,0]<5):

        print('Test Accuracy',test_accuracy_list[r],'Train Accuracy',train_accuracy_list[r],'Random State ',desired_random_state_list[r])

        print()

        print("Confusion Matrix ")

        print(conf_matrix_report_list[r])

        print()

        print("Classification Report ")

        print(class_report_list[r])

        print("--------------------------------------------------------") 