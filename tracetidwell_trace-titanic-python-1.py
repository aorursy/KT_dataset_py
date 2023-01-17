import pandas as pd

import csv as csv

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from time import time



train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



data = np.array(train)

test_data = np.array(test)
# The size() function counts how many elements are in

# in the array and sum() (as you would expects) sums up

# the elements in the array.



number_passengers = np.size(data[0::,1].astype(np.float))

number_survived = np.sum(data[0::,1].astype(np.float))

proportion_survivors = number_survived / number_passengers
women_only_stats = data[0::,4] == "female" # This finds where all 

                                           # the elements in the gender

                                           # column that equals “female”

men_only_stats = data[0::,4] != "female"   # This finds where all the 

                                           # elements do not equal 

                                           # female (i.e. male)
# Using the index from above we select the females and males separately

women_onboard = data[women_only_stats,1].astype(np.float)     

men_onboard = data[men_only_stats,1].astype(np.float)



# Then we finds the proportions of them that survived

proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)  

proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard) 



# and then print it out

print('Proportion of women who survived is %s' % proportion_women_survived)

print('Proportion of men who survived is %s' % proportion_men_survived)
prediction = test_data[0::,3] == "female"

prediction = prediction * 1



submission = pd.DataFrame({"PassengerId": test_data[0::, 0], "Survived": prediction})



submission.to_csv("FirstAttempt.csv", index=False)