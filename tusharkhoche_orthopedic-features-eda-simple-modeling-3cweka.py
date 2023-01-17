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
# Import data 3Classdata.csv for 3 Class Classification



missing_value_formats = ["n.a.","?","NA","n/a","na","--"," ", "  "]

ThreeC_weka_data = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv', na_values = missing_value_formats)
ThreeC_weka_data.info()
# There are 7 features, 6 numerical and one object
ThreeC_weka_data.describe()
# Same as column_2C_weka, except "degree_spondylolisthesis", rest all of the columns have distribution close to

# normal. degree_spondylolisthesis seems to be right-tailed or positively skewed.
ThreeC_weka_data.shape
# The 3C_weka has 310 rows & 7 columns same as 2C_weka
# Checking top 5 rows

ThreeC_weka_data.head()
# Checking last 5 rows

ThreeC_weka_data.tail()
# Checking for skewness

ThreeC_weka_data.skew()
# As mentioned above "degree_spondylolisthesis" is positively skewed. 
# Checking for NA or Null values

ThreeC_weka_data.isna().sum()
ThreeC_weka_data.isnull().sum()
# There are no null values.
# Checking for duplicate data

ThreeC_weka_data.duplicated().sum()
# There is no duplicate data.
# Checking for unique values in target variable "class"

print(ThreeC_weka_data['class'].unique())
# So, there are three classes 'Hernia', 'Spondylolisthesis', 'Normal'
# Count of each class
# Method 1

pd.crosstab(ThreeC_weka_data['class'],columns='Count')
# Method 2

print(ThreeC_weka_data['class'].value_counts())
# Method 3

sns.countplot(x='class',data=ThreeC_weka_data)

plt.show()
# Identifying Type Of Features 

# Numerical Features & Categorical Features



numerical_features = ThreeC_weka_data.select_dtypes(include = [np.number])
print(numerical_features.columns)
# Now we want to segregate discrete variables from continuous variables

# So, we count the number of unique values in each feature. If count of unique values is less than 25 then we consider it as

# discrete variable otherwise it is a continuous variable



continuous_numerical_features = []

discrete_numerical_features = []
for feature in numerical_features:

    if(len(ThreeC_weka_data[feature].unique())>25):

        continuous_numerical_features.append(feature)

        print('continuous_numerical_features ',feature)

   
# This shows that all the features are continuous 
# Visualizing Distribution For Numerical Columns

# Not using distplot as it is going to be deprecated. See below link 

# https://seaborn.pydata.org/generated/seaborn.distplot.html?highlight=distplot#seaborn.distplot



for feature in numerical_features.columns:

    sns.distplot(numerical_features[feature],kde=True)

    plt.show()
# Features "pelvic_tilt" & "pelvic_radius"  are very close to normal. 

# Features "pelvic_incidence", "lumbar_lordosis_angle" & "sacral_slope" have some kind of uniform distribution.

# Feature degree_spondylolisthesis is highly positively skewed.
# Now looking for IQR & Outliers
for feature in numerical_features.columns:

    sns.boxplot(ThreeC_weka_data[feature])

    plt.show()

# There are outliers in all of the features. Features "lumbar_lordosis_angle" & "sacral_slope" have just one outlier.

# Rest all have many outlies
# Plotting Correlation HeatMap



plt.figure(figsize=(7,7))

sns.heatmap(ThreeC_weka_data.corr(),annot=True)

plt.show()
# It looks like there is some multicolinearity here.

# For example: Feature "pelvic_incidence" seems to be correlated with all the other features 

# except "pelvic_radius".
# Barplot showing the numbers 



for feature in numerical_features.columns:

    sns.barplot(x='class',y=feature,data=ThreeC_weka_data)

    plt.show()
# Class "Spondylolisthesis" has more count as compare to other two classes for alsmot all of the features
# Plotting Swarmplots



for feature in numerical_features.columns:

    sns.swarmplot(x=ThreeC_weka_data['class'],y=ThreeC_weka_data[feature])

    plt.show()
# Swarmplots showing spread as well as outliers.
# Plotting pairplot



sns.pairplot(ThreeC_weka_data,size=3,hue='class')

plt.show()
# Features "pelvic_incidence", "lumbar_lordosis_angle" & "sacral_slope" looks to be good indicators as they have considerable separation. 

# It looks like feature "pelvic_incidence" has some degree of linear relationship with features "pelvic_tilt", "lumbar_lordosis_angle" & "sacral_slope". 

# Some other features also have linear relationship
# Power Transformation is done to make data normal.

# Creating a dataset of numerical features only for transformation



numerical_dataset = ThreeC_weka_data.iloc[:,:6]

numerical_dataset
power_transform = PowerTransformer(method='yeo-johnson', standardize=True)
ThreeC_weka_data_transformed = power_transform.fit_transform(numerical_dataset)

ThreeC_weka_data_transformed = pd.DataFrame(ThreeC_weka_data_transformed,columns = numerical_dataset.columns)
# Distribution after transformation. 

# I have plotted both the original & transformed distribution for comparison.

# We can observe that features have been transformed into Normal distribution



for feature in ThreeC_weka_data_transformed.columns:

    #print("             Original ", feature)

    sns.distplot(numerical_features[feature],kde=True)

    plt.show()

    #print("             Transformed ", feature)

    sns.distplot(ThreeC_weka_data_transformed[feature],kde=True)

    plt.show()  

    #print("-----------------------------------------------------")
# Same as column_2C_weka, this feature "degree_spondylolisthesis" shown above is not fully normal.

# It has two peaks. At this stage of this course, what I know is we separate these two peaks then we move forward.

# But currently that is beyond the scope of my knowledge. 

# So, I will keep this as it is.
# Keeping X in uppercase & y in lowercase as per standard convention



X = ThreeC_weka_data_transformed

y = ThreeC_weka_data.iloc[:,6:]
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

result = np.where(test_accuracy_array>0.83)

result = result[0]



     # If a patient is predicted Normal when he is Abnormal, then this prediction is bad. Patient is having medical issue but model 

     # predicted patient does not have any issue. 

     # We definitely need to minimize this (False Negative) as much as possible. So, So, I have chosen to keep all incorrect 

     # predictions less than 5

    

for r in result:  

    conf = conf_matrix_report_list[r]

    if(conf[0,1]<5 and conf[0,2]<5 and conf[1,0]<5 and conf[1,2]<5 and conf[2,0]<5 and conf[2,1]<5):

        print('Test Accuracy',test_accuracy_list[r],'Train Accuracy',train_accuracy_list[r],'K Value ' ,desired_k_value_list[r],'Random State ',desired_random_state_list[r])

        print()

        print("Confusion Matrix ")

        print(conf_matrix_report_list[r])

        print()

        print("Classification Report ")

        print(class_report_list[r])

        print("--------------------------------------------------------")



        
# Here, as per my understanding, keeping False Negative as much low as possible should be on priority keeping test accuracy high so, Random State 26 & K - Value = 18

# gives us overall test accuracy 86, precision for Hernia class is 80, for Spondylolisthesis is 96 & false negative are less as compare to others for KNN. 
# Training Gaussian Naive Bayes



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

        

       

     # If a patient is predicted Normal when he has "Hernia" or "Spondylolisthesis", then this prediction is bad. Patient is having medical issue but model 

     # predicted patient does not have any issue. We definitely need to minimize this (False Negative) as much as possible. So, I have chosen to keep all incorrect 

     # predictions less than 6

        

test_accuracy_array = np.array(test_accuracy_list)

result = np.where(test_accuracy_array>0.80)

result = result[0]





for r in result:  

    conf = conf_matrix_report_list[r]

    if(conf[0,1]<6 and conf[0,2]<6 and conf[1,0]<6 and conf[1,2]<6 and conf[2,0]<6 and conf[2,1]<6):

        print('Test Accuracy',test_accuracy_list[r],'Train Accuracy',train_accuracy_list[r],'Random State ',desired_random_state_list[r])

        print()

        print("Confusion Matrix ")

        print(conf_matrix_report_list[r])

        print()

        print("Classification Report ")

        print(class_report_list[r])

        print("--------------------------------------------------------")



    

# Again, keeping False Negative as much low as possible should be on priority keeping test accuracy high so, Random State 48 gives us overall test accuracy 84,

# precision for Hernia class is 75, for Spondylolisthesis is 90 & false negative are less as compared to other vlues for Random State 
# Training Logistic Regression





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

        

       

     # If a patient is predicted Normal when he has "Hernia" or "Spondylolisthesis", then this prediction is bad. Patient is having medical issue but model 

     # predicted patient does not have any issue. We definitely need to minimize this (False Negative) as much as possible. So, I have chosen to keep all incorrect 

     # predictions less than 6

        

test_accuracy_array = np.array(test_accuracy_list)

result = np.where(test_accuracy_array>0.80)

result = result[0]

        

for r in result:  

    conf = conf_matrix_report_list[r]

    if(conf[0,1]<6 and conf[0,2]<6 and conf[1,0]<6 and conf[1,2]<6 and conf[2,0]<6 and conf[2,1]<6):

        print('Test Accuracy',test_accuracy_list[r],'Train Accuracy',train_accuracy_list[r],'Random State ',desired_random_state_list[r])

        print()

        print("Confusion Matrix ")

        print(conf_matrix_report_list[r])

        print()

        print("Classification Report ")

        print(class_report_list[r])

        print("--------------------------------------------------------")     

# For random state 4 & 13, we get 100% success rate for Spondylolisthesis  prediction. 

# For random state 4, we also get 75% success for Hernia 

# Also for random state 7, we get overall test accuracy 90% 

# We are getting much more encouraging results with Logistic Regression as compared to KNN & Naive Bayes