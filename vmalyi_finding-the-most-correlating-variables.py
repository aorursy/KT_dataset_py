import pandas as pd



# Always display all the columns

pd.set_option('display.width', 5000) 

pd.set_option('display.max_columns', 60) 



dataset = pd.read_csv("../input/kc_house_data.csv") # read the dataset



dataset.head(5) # print 5 first rows from the dataset
dataset.dtypes # get an overview of data types presented in the dataset
print(dataset.isnull().any())
# let's observe unique values presented in potentially categorical columns

print("bedrooms")

print(sorted(dataset.bedrooms.unique()))

print("bathrooms")

print(sorted(dataset.bathrooms.unique()))

print("floors")

print(sorted(dataset.floors.unique()))

print("waterfront")

print(sorted(dataset.waterfront.unique()))

print("view")

print(sorted(dataset.view.unique()))

print("condition")

print(sorted(dataset.condition.unique()))

print("grade")

print(sorted(dataset.grade.unique()))
# Create new categorical variables

dataset['waterfront'] = dataset['waterfront'].astype('category',ordered=True)

dataset['view'] = dataset['view'].astype('category',ordered=True)

dataset['condition'] = dataset['condition'].astype('category',ordered=True)

dataset['grade'] = dataset['grade'].astype('category',ordered=False)



# Remove unused variables

dataset = dataset.drop(['id', 'date'],axis=1)



dataset.dtypes # re-check data types in the dataset after conversion above
dataset['basement_is_present'] = dataset['sqft_basement'].apply(lambda x: 1 if x > 0 else 0)

dataset['basement_is_present'] = dataset['basement_is_present'].astype('category', ordered = False)



dataset['is_renovated'] = dataset['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

dataset['is_renovated'] = dataset['is_renovated'].astype('category', ordered = False)



dataset.dtypes
from scipy import stats



CATEGORICAL_VARIABLES = ["waterfront", 

                       "basement_is_present", 

                       "is_renovated", 

                       "bedrooms", 

                       "bathrooms", 

                       "floors", 

                       "view", 

                       "condition",

                       "grade"]



for c in CATEGORICAL_VARIABLES:

    if c not in ["waterfront", "basement_is_present", "is_renovated"]:

        correlation = stats.pearsonr(dataset[c], dataset["price"])

    else:

        correlation = stats.pointbiserialr(dataset[c], dataset["price"])

    print("Correlation of %s to price is %s" %(c, correlation))
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



CONTINUOUS_VARIABLES = ["price", 

                       "sqft_living", 

                       "sqft_lot", 

                       "sqft_above", 

                       "sqft_basement", 

                       "yr_built", 

                       "yr_renovated", 

                       "zipcode",

                       "lat",

                       "long",

                       "sqft_living15",

                       "sqft_lot15"]



# create new dataframe containing only continuous variables

cont_variables_dataframe = dataset[CONTINUOUS_VARIABLES]

# calculate correlation for all continuous variables

cont_variables_correlation = cont_variables_dataframe.corr()



# plot the heatmap showing calculated correlations

plt.subplots(figsize=(11, 11))

plt.title('Pearson Correlation of continous features')

ax = sns.heatmap(cont_variables_correlation, 

                 annot=True, 

                 linewidths=.5, 

                 cmap="YlGnBu",

                 square=True

                );