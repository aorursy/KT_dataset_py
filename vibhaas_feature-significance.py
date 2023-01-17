from IPython.display import Image

import os

Image("../input/employee/Feature Significance.JPG")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import modules



#import libraries for data handling

import os

import pandas as pd # for dataframes

import numpy as np



#import for visualization

import seaborn as sns # for plotting graphs

import matplotlib

import matplotlib.pyplot as plt # for plotting graphs

%matplotlib inline



#import for Linear regression

from sklearn.linear_model import LinearRegression



#import warnings

import warnings

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore', category=DeprecationWarning)

basetable1 = pd.read_excel('/kaggle/input/employee/employee_churn.xls')

basetable1.head(5)
# correcting column names

basetable1 = basetable1.rename(columns={'left': 'target', 'Work_accident': 'work_accident'})

basetable1 = basetable1.rename(columns={'Departments ': 'departments'})

basetable1.info()
basetable1.head()
# Create the dummy variable

dummies_salary = pd.get_dummies(basetable1["salary"], drop_first = True)



# Add the dummy variable to the basetable

basetable2 = pd.concat([basetable1, dummies_salary], axis = 1)



# Delete the original variable from the basetable

del basetable2["salary"]



# Create the dummy variable

dummies_departments = pd.get_dummies(basetable2["departments"], drop_first = True)





# Add the dummy variable to the basetable

basetable2 = pd.concat([basetable2, dummies_departments], axis = 1)



# Delete the original variable from the basetable

del basetable2["departments"]





basetable2.head()
# chck if there are any NUll values

basetable2.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="whitegrid")





from sklearn.ensemble import RandomForestClassifier



#modeling_data



# Machine Learning 

FI_predictor = basetable2.drop(["target"], 1)

FI_target = basetable2["target"]



clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')

clf = clf.fit(FI_predictor, FI_target)



#have a look at the importance of each feature.

features = pd.DataFrame()

features['feature'] = FI_predictor.columns

features['importance'] = clf.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

##features.set_index('feature', inplace=True)





# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(10, 10))



# Load the dataset

FI = features.sort_values("importance", ascending=False)



# Plot the total crashes

sns.set_color_codes("pastel")

sns.barplot(x="importance", y="feature", data=FI,

            label="importance", color="b")





# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 0.4), ylabel="",

       xlabel="Feature Importance")

sns.despine(left=True, bottom=True)
# Checking absolute values of Feature importance

FI
# Import the GB and accuracy modules

#Import Gradient Boosting Classifier model

from sklearn.ensemble import GradientBoostingClassifier

import sklearn.metrics as metrics

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score 

from sklearn.metrics import mean_absolute_error, accuracy_score



# define auc function

def gbacc(variables, target, basetable):

    predictions = 0 # reset the value

    auc= 0 # reset the value

    

    X = basetable[variables]

    Y = basetable[target]

    #Create Gradient Boosting Classifier

    gb = GradientBoostingClassifier()

    gb.fit(X, Y)

    predictions = gb.predict(X)

    gb_acc = metrics.accuracy_score(Y, predictions)

    return(gb_acc)

gb_acc = gbacc(["satisfaction_level", "promotion_last_5years","number_project"],["target"], basetable2)

print(round(gb_acc,2))
def next_best_v(current_variables,candidate_variables, target, basetable):

    best_gbacc = -1

    best_variable = None

    

	# Calculate the auc score of adding v to the current variables

    for v in candidate_variables:

        gbacc_v = gbacc(current_variables + [v],target, basetable)

        

		# Update best_auc and best_variable adding v led to a better auc score

        if gbacc_v >= best_gbacc:

            best_gbacc = gbacc_v

            best_variable = v

            

    return best_variable

next_variable = next_best_v(["satisfaction_level","last_evaluation"], ["number_project","average_montly_hours","time_spend_company","work_accident","promotion_last_5years","low","medium","RandD","accounting","hr","management","marketing","product_mng","sales","support","technical"], ["target"], basetable2)

print(next_variable)





candidate_variables = ["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","work_accident","promotion_last_5years","low","medium","RandD","accounting","hr","management","marketing","product_mng","sales","support","technical"]

current_variables = []

target = ["target"]



max_number_variables = 8

number_iterations = min(max_number_variables, len(candidate_variables))

result = []

for i in range(0,number_iterations):

    

    next_variable = next_best_v(current_variables,candidate_variables,target,basetable2)

    

    current_variables = current_variables + [next_variable]

    candidate_variables.remove(next_variable)

    #result.append((gb_acc))

##print(gb_acc)

print(current_variables)



srt_variable = ["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","work_accident","promotion_last_5years","low","medium","RandD","accounting","hr","management","marketing","product_mng","sales","support","technical"]

current_variables = []

target = ["target"]

result = []



for v in srt_variable:

        gbacc_v = gbacc(current_variables + [v],target, basetable2)

        

        #print(gbacc_v)

        #print(v)

        result.append((v,gbacc_v))

        



pd.DataFrame(result)

df = pd.DataFrame(result, columns =['Variable', 'Model_Accuracy']) 

df   
import matplotlib.pyplot as plt





sns.set(style="whitegrid")

sns.set_color_codes("pastel")



df1 = df.sort_index(axis = 1)

ax = sns.catplot(x="Variable", y="Model_Accuracy",kind = "point",  data=df1)



plt.xticks(rotation=90)

    

# Show plot

plt.show()