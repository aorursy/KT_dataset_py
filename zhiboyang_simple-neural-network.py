import numpy as np

import pandas as pd
#Sigmoid funciont



def nonlin(x, deriv = False):

    if (deriv == True):

       return x * (1 - x)

    else:

       return 1/(1 + np.exp(-x))
#Split dataset into training dat and test data



all_data = pd.read_csv('../input/voice.csv')

features = [feat for feat in all_data.columns if feat != 'label']

output = 'label'

num_datapoints = len(all_data)



#Training data is first 90% of male and female

#Test data is las 10% of male and female



test_male = int(1584 * 0.1)

test_female = int((num_datapoints-1584) * 0.1)



all_data[output].replace(['male','female'], [1, 0], inplace = True)



train_data_male = all_data[features].iloc[0:(1583 - test_male)].values

train_data_female = all_data[features].iloc[1584:-test_female].values

test_data_male = all_data[features].iloc[(1583 - test_male):1583].values 

test_data_female = all_data[features].iloc[-test_female:].values



train_data = np.concatenate((train_data_male, train_data_female), axis = 0)

test_data = np.concatenate((test_data_male, test_data_female),axis = 0)



train_labels_male = all_data[output].iloc[0:(1583 - test_male)].values

train_labels_female = all_data[output].iloc[1584:-test_female].values

test_labels_male = all_data[output].iloc[(1583 - test_male):1583].values 

test_labels_female = all_data[output].iloc[-test_female:].values



train_labels = np.concatenate((train_labels_male, train_labels_female), axis = 0)

test_labels = np.concatenate((test_labels_male, test_labels_female), axis = 0)



print(len(train_data))

print(len(test_data))
#To ensure random number of every loop is the same

np.random.seed(1)

#Initialize weights

syn0 = 2 * np.random.random((20,1)) - 1

l0 = train_data

#Keep updating weights

for iter in range(100000):

    l0 = train_data

    #Input(l0) with weights(syn0) 

    #Combined by sigmoid function

    l1 = nonlin(np.dot(l0,syn0))

    

    #How different that l1 from exact labels

    l1_error = train_labels.T - l1.T

    

    #Updating weight

    syn0 += np.dot(l0.T, l1_error.T)

    
count_0 = 0

count_1 = 0

for i in l1:

    if 0 in i:

       count_0 += 1

    else:

       count_1 += 1

print("predicted number of female", count_0)

print("predicted number of male", count_1)

print("total number of train data:",count_0 + count_1)
correct_predic = 0

wrong_predic = 0

for i in range(0,2851):

    if train_labels[i] == l1[i]:

       correct_predic += 1

    else:

       wrong_predic += 1

    

print("Number of correct prediction", correct_predic)

print("Number of wrong prediction", wrong_predic)
print("Correct rate is: ")

print(correct_predic/(wrong_predic+correct_predic))
#Test part



l0 = test_data

   

l1 = nonlin(np.dot(l0,syn0))

    

    
count_0 = 0

count_1 = 0

for i in l1:

    if 0 in i:

       count_0 += 1

    else:

       count_1 += 1

print("predicted number of female", count_0)

print("predicted number of male", count_1)

print("total number of train data:",count_0 + count_1)
correct_predic = 0

wrong_predic = 0

for i in range(0,316):

    if test_labels[i] == l1[i]:

       correct_predic += 1

    else:

       wrong_predic += 1

    

print("Number of correct prediction", correct_predic)

print("Number of wrong prediction", wrong_predic)
print("Correct rate is: ", correct_predic/(wrong_predic + correct_predic))