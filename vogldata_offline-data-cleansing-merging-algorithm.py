

""" This contribution is submitted by Markus Vogl of Markus Vogl {Business & Data Science}
as part of the lecture "Data Science for algorithmic financial markets & time-series analysis" 
at the University of Applied Sciences Aschaffenburg Germany
(Technische Hochschule Aschaffenburg-Deutschland). 
Markus Vogl is part of the Behavioral Finance & Accounting Lab and a PhD candidate
in financial & risk modelling and Chaos Theory.
Please note that the contribution to kaggle will not be all encompassing and at a lower level
since the major aim is to focus on basic applications of ML & Data Science for the lecture. 
Nevertheless, we try to contribute a respective level of data analysis.
Note, that the rights of the course and the code belongs to Markus Vogl. 

You can visit the Lab at: 
https://www.th-ab.de/ueber-uns/organisation/labor/behavioral-accounting-and-finance-lab/

Our Data Science operation at: 
https://vogl-datascience.de/

Our full lecture on YouTube at:
https://www.youtube.com/playlist?list=PLFXw4NpfUWMi4enJ2_jtKjSwHPhnNbd2G

Further, the lecture and code fall under an CC-BY-NC-SA 3.0 DE license.

If you are interested in our research, please contact us anytime. 

This is a Python 3 environment."""

#Import relevant packages
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from tabulate import tabulate as tabu 
from functools import reduce

#set path: Please note that we will not loop through the files to save lines of code, 
#due to the capability of our students to replicate the results.
t_path = "../input/fatal-police-shootings-in-the-us/"

#Import relevant data sets: Please note, that we will use the raw data sets as provided 
#via the kaggle download: https://www.kaggle.com/kwullum/fatal-police-shootings-in-the-us

t_1 = pd.read_csv(t_path + "MedianHouseholdIncome2015.csv", 
                  sep=",", encoding="unicode_escape")

t_2 = pd.read_csv(t_path + "PercentagePeopleBelowPovertyLevel.csv", 
                  sep=",", encoding="unicode_escape")

t_3 = pd.read_csv(t_path + "PercentOver25CompletedHighSchool.csv", 
                  sep=",", encoding="unicode_escape")

t_4 = pd.read_csv(t_path + "ShareRaceByCity.csv", 
                  sep=",", encoding="unicode_escape")

""" Merge t_1 to t_4 into one df:
(1) Define a list of all dfs to be merged
(2) use reduce function and lambda statement to conduct inner join based on City column
(3) drop redundant columns 
(4) rename some columns for consistency"""

t_data_frames = [t_1, t_2, t_3, t_4]

df_living_information_data = reduce(lambda left,right: pd.merge(left,right,
                                    on=['City'],how='inner'),
                                    t_data_frames)

df_living_information_data.drop(
                                ["Geographic Area_y","Geographic Area", 
                                 "Geographic area"], 
                                axis=1, inplace=True)

df_living_information_data.rename(columns={ #beginning of dict
                                            "Geographic Area_x":"geographic_area",
                                            "Median Income":"median_income", 
                                            "City":"city"
                                            }, #end of dict
                                            inplace=True)    

# print the heads of the two dfs: We see the raw data is not consistent, since 
# several cities display different other parameter realisations. 
#print(tabu(df_living_information_data.head(), missingval="?", tablefmt="simple",
#          headers=df_living_information_data.columns), "\n", "\n")
#print(tabu(df_police_killings_data.head(), missingval="?", tablefmt="simple", 
#          headers=df_police_killings_data.columns), "\n", "\n")

""" We will now count cities and will replace the duplicates via the mean values 
of all delivered values. Since we do not have an extraction date,we cannot 
determine the latest entry dates.
We do it as follows:
(1) Create combined string of geographic_area and city to ensure doubles are 
really doubles and not same named cities in different areas
(2) check for missing values
(3) create unique city_area combinations in new df
(4) split the combination again
(5) fill in the mean values of the doubles"""

#create (1)
df_living_information_data["cat_city"] = df_living_information_data["geographic_area"] + df_living_information_data["city"]

#display boolean matrix, where Nulls are True == 1 and values are False == 0: We only have one missing household income;
#therefore we drop the row accordingly.
#sns.heatmap(df_living_information_data.isnull(), cmap="viridis")
#plt.show()

#drop NaNs as in (2)
df_living_information_data.dropna(axis=0,inplace=True)

# Since no row is complete (e.g. (X) blanks)!!! We cannot solve the issue with this approach:
#df_living_information_data = df_living_information_data[df_living_information_data.applymap(np.isreal).all(1)]
#print(df_living_information_data.head(25), "\n", len(df_living_information_data))

# check max Length of geographic_area IDs and create (3)
#print(max(df_living_information_data["geographic_area"].str.len()))
t_cat_city = list(dict.fromkeys(df_living_information_data["cat_city"].to_list()))
df_cleansed_living_information_data = pd.DataFrame(columns=df_living_information_data.columns, dtype=float)

# fill new df with unique values before calculation of means and set cat_city2 as new index of df
df_cleansed_living_information_data["cat_city"] = t_cat_city
df_cleansed_living_information_data["cat_city2"] = t_cat_city
df_cleansed_living_information_data["geographic_area"] = df_cleansed_living_information_data["cat_city"].str[:2]
df_cleansed_living_information_data["city"] = df_cleansed_living_information_data["cat_city"].str[2:]
df_cleansed_living_information_data= df_cleansed_living_information_data.set_index(["cat_city2"])

#since the dataset is inconsistent, redundant and has wrong values ("shame to the US departements...poor job!")
# we clean the data on city-level! meaning, we take every cat_city and erase the errors and calculate the mean
# of every column as part of feature engineering.

t_cols = df_living_information_data.columns
#print(t_cols)
print("Loop started.")

# each loop picks a column and calculates the means and appends it to the cleaned df
for cols in range(2,len(t_cols)-1):

    t_dict = {}
  
    t_df = pd.DataFrame(columns=["cat_city",str(t_cols[cols])], dtype=float)
    t_df["cat_city"] = df_living_information_data["cat_city"]
    t_df[str(t_cols[cols])] = df_living_information_data[str(t_cols[cols])]

    for cat_city in df_cleansed_living_information_data["cat_city"]:

        if len(t_df["cat_city"]) > 1:

            tt_df = t_df[t_df["cat_city"]==cat_city]
            t_errors = (tt_df == "(X)") | (tt_df == "-") | (tt_df == "2,500-") | (tt_df == "250,000+")
            tt_df = tt_df[t_errors==False]
            
            tt_df[str(t_cols[cols])] = pd.to_numeric(tt_df[str(t_cols[cols])]).mean()
            t_dict[str(cat_city)] = tt_df[str(t_cols[cols])].unique()

    t_dict_series = pd.DataFrame.from_dict(t_dict, orient="index")
    df_cleansed_living_information_data[str(t_cols[cols])]= t_dict_series[0]

# Now we save the cleansed data for further processing in an excel notebook.
# Code will continue in script file 2. 
df_cleansed_living_information_data.to_excel("../input/output/df_cleansed.xlsx", 
                                            sheet_name='US_Shootings', index = True)

print("Completed!")