import numpy as np 

import pandas as pd 



full_df= pd.read_csv('../input/ashrae-global-thermal-comfort-database-ii/ashrae_db2.01.csv', low_memory= False)



#create smaller dataframe with only the person-specific features, non-person-specific features, and what we're trying to predict

personal_feats= ['Age', 'Sex', 'Clo','Met',"Subject«s height (cm)", "Subject«s weight (kg)"]

other_feats= ['Year', 'Season', 'Koppen climate classification', 'Climate', 'City', 'Country', 'Building type', 'Cooling startegy_building level', 'Heating strategy_building level','Air temperature (C)', 'Outdoor monthly air temperature (C)']

label= ['Thermal preference']

df= full_df[personal_feats + other_feats + label]

df= df.dropna()

print('Dataset size: ', len(df))

df.head()
from sklearn.model_selection import train_test_split



#define features and labels

feats= df.loc[:, df.columns != label[0]]

labs= df.loc[:, df.columns == label[0]]



#create 90/10 train/test split

x_train, x_test, y_train, y_test= train_test_split(feats, labs, test_size= 0.1, random_state= 0)



#see how many different levels there are for each categorical feature

categ_feats= feats.select_dtypes(include= ['category', object]).columns

for i in categ_feats:

    print(i, df[i].nunique(), '\n')  
from sklearn.preprocessing import OneHotEncoder



encoder= OneHotEncoder(handle_unknown= 'ignore', sparse= False)



OH_feats= pd.DataFrame(encoder.fit_transform(feats[categ_feats])) #encode the categorical features

OH_feats.columns = encoder.get_feature_names(categ_feats) #ensure encoded col names are meaningful

OH_feats.index= feats.index #indices need to match in order to add one-hot encodings to original dataframe

feats= feats.drop(categ_feats, axis= 1) #delete categorical cols from original dataframe

feats= pd.concat([feats, OH_feats], axis= 1) 



#create 90/10 train/test split

x_train, x_test, y_train, y_test= train_test_split(feats, labs, test_size= 0.1, random_state= 0)
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel



def important_feats(x_train, y_train, x_test):

    """

    Function that fits a random forest and extracts the importance weight from the forest for each feature to determine which features are most important

    (Features with an importance weight greater than the median weight are most important)

    

    INPUTS: x_train is a pandas dataframe where each row is one example and each column is a feature (training data)

    y_train is a pandas dataframe with the corresponding labels to each example in x_train

    x_test is a pandas dataframe where each row is one example and each column is a feature (test data)

    

    OUTPUTS: x_train_new is the same as x_train except with only the most important features retained

    x_test_new is the same as x_test except with only the most important features retained



    """

    #define and fit tree

    forest= RandomForestClassifier(n_estimators= 1000, random_state= 0)

    forest.fit(x_train, np.ravel(y_train))



    #select most important features

    selector= SelectFromModel(forest, threshold= 'median')

    selector.fit(x_train, np.ravel(y_train))

    important_feats= np.array([]) #store the names of the most important features

    for i in selector.get_support(indices= True):

        important_feats= np.append(important_feats, x_train.columns[i])

    

    #return only the most important features (for both training and test sets)

    x_train_new= pd.DataFrame(selector.transform(x_train), columns= important_feats)

    x_test_new= pd.DataFrame(selector.transform(x_test), columns= important_feats)

    

    return important_feats, x_train_new, x_test_new





#redefine the columns that are person-specific features (names are different now because of the one-hot encoding!)

personal_feats= ['Sex_Female', 'Sex_Male', 'Age', 'Clo', 'Met', 'Subject«s height (cm)', 'Subject«s weight (kg)']



#for forest that uses only person-specific features:

x_train_personal= x_train.loc[:, personal_feats]

x_test_personal= x_test.loc[:, personal_feats]



#identify the most important person-specific features:

personal_important_feats, x_train_personal_new, x_test_personal_new= important_feats(x_train_personal, y_train, x_test_personal)

print(personal_important_feats)
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import balanced_accuracy_score



def train_eval_knn(x_train, y_train, x_test, y_test):

    """

    Function that trains and tests a kNN multi-class classifier and returns the average recall on the test set

    

    INPUTS: x_train and x_test are 2D numpy arrays where each row is one example and each column is a feature,

    y_train and y_test are pandas dataframes with the corresponding label for the examples in x_train/x_test

    OUTPUT: test_recall is the average recall for the test set (single float value)

    """

    knn= KNeighborsClassifier()

    knn.fit(x_train, np.ravel(y_train))

    #note: default settings on kNN uses the Euclidean distance as the distance metric and equally weights all examples

    

    test_predicts= knn.predict(x_test)

    test_recall= balanced_accuracy_score(y_test, test_predicts)

    

    return test_recall



#scale the training data so that all features have a mean of 0 and unit variance

scaler= StandardScaler()

x_train_personal_new= scaler.fit_transform(x_train_personal_new)



#scale the test data using the mean and variance calculated from the training data

x_test_personal_new= scaler.transform(x_test_personal_new)



#train and test kNN that uses the most important person-specific features

personal_recall= train_eval_knn(x_train_personal_new, y_train, x_test_personal_new, y_test)

print(personal_recall)
#redefine the columns that are non-person-specific features (names are different now because of the one-hot encoding!)

other_feats= set(x_train.columns) - set(personal_feats)



#get training and test training sets that have only the non-person-specific features

x_train_other= x_train.loc[:, other_feats]

x_test_other= x_test.loc[:, other_feats]



#identify the most important non-person-specific features:

other_important_feats, x_train_other_new, x_test_other_new= important_feats(x_train_other, y_train, x_test_other)

print(other_important_feats)



#scale feature values before running kNN...

x_train_other_new= scaler.fit_transform(x_train_other_new)

x_test_other_new= scaler.transform(x_test_other_new)



#train and test kNN that uses the most important non-person-specific features

other_recall= train_eval_knn(x_train_other_new, y_train, x_test_other_new, y_test)

print(other_recall)
#identify the most important features:

important_feats, x_train_new, x_test_new= important_feats(x_train, y_train, x_test)

print(important_feats)



#scale feature values before running kNN...

x_train_new= scaler.fit_transform(x_train_new)

x_test_new= scaler.transform(x_test_new)

x_train= scaler.fit_transform(x_train)

x_test= scaler.fit_transform(x_test)



#train and test kNN that uses the most important features

recall= train_eval_knn(x_train_new, y_train, x_test_new, y_test)

print(recall)



#train and test kNN that uses all of the features, not just the most important ones

recall_allfeats= train_eval_knn(x_train, y_train, x_test, y_test)

print(recall_allfeats)