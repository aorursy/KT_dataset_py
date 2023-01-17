#Place your import here

import pandas as pd

import numpy as np



def ReadData():

    df_salaries = None

    df_names = None

    

    salaries_file_path = "../input/Salaries.csv"

    names_file_path = "../input/Names.csv"

    

    df_salaries = pd.read_csv(salaries_file_path)

    df_names = pd.read_csv(names_file_path)

    return [df_salaries, df_names]
[df_salaries, df_names] = ReadData()

assert df_names.shape == (24713, 5)
#use this to review the data

df_names.head()
for index,row in df_names.head().iterrows():

    print(row)
[df_salaries, df_names] = ReadData()

assert df_salaries.shape == (27386, 6)
df_salaries.head()
for index,row in df_salaries.head().iterrows():

    print(row[2])
def ParseNames(df_names):

    """

    

    INPUT: the pandas dataframe contains names.csv

    

    OUTPUT: two dictionaries: male_names, female_name.

    The key in each of these dictionaries will be names 

    (in all lowercase)and the value will be the sum of the 

    counts for the given name when it applies to the given gender.

    

    USE ONLY ITERROWS(), NO GROUPING OR FILTERING YET! 

    This above function will take a minute or two to run. 

    """

    

    #Initialize empty dictionaries for names

    male_names = {}

    female_names = {}

    

    for index,row in df_names.iterrows():

        name_lowercase = row[1].lower()

        gender = row[3]

        count = row[4]

        if gender == 'F':

            female_names[name_lowercase] = count

        else:

            male_names[name_lowercase] = count

        

    return male_names, female_names

[male_names, female_names] = ParseNames(df_names)

assert len(male_names) == 9482

assert len(female_names) == 15231
def GetFirstName(name):

    

    """

    Gets the first name from a name in the column

    EmployeeName in Salaries.csv.

    INPUT: name as string

    OUTPUT: first name in all lowercase

    """

    first_name = ""



    first_name = name.split(" ")[0].lower()

    

    return first_name
assert GetFirstName("Dennis Zhang") == "dennis"
def AddGender(first_name, male_names, female_names):

    

    """

    Find the most likely gender associated with a first name.

    

    INPUT: first_name, males_names and females_names which are the dictionaries 

    returned from ParseNames().

    

    OUTPUT:

    "M" if male_names[name] > female_names[name]

    

    "F" if male_names[name] <= female_names[name]

    

    "NaN" if the name doesn't apper in either dictionary

    """

    

    return_gender = "NA"

#     print(first_name)

    male_count = male_names[first_name] if first_name in male_names else 0

    female_count = female_names[first_name] if first_name in female_names else 0

#     print(male_count,female_count)

    if(male_count>female_count):

        return_gender = "M"

    elif(male_count<=female_count and female_count!=0):

        return_gender = "F"

    

    return return_gender
[df_salaries, df_names] = ReadData()

assert AddGender("charles", male_names, female_names) == "M"

assert AddGender("jasmine", male_names, female_names) == "F"

assert AddGender("dennis", male_names, female_names) == "M"
def AddGenderToDF(df_salaries, male_names, female_names):

    """

    This function will return a new dataframe with two new columns

    on top of the existing columns in df_salaries. 

    

    The first column is called "first_name" which contains the first

    name of the person.

    

    The second column is called "gender" which contains the gender

    inforamtion of the person from the AddGender() function.

    """

    first_names = []

    genders = []

    for index,row in df_salaries.iterrows():

        name_lowercase = row[2].lower()

        first_name = GetFirstName(name_lowercase)

        first_names.append(first_name)

        gender = AddGender(first_name,male_names,female_names)

        genders.append(gender)

    

    df_salaries.insert(0,"first_name",first_names)

    df_salaries.insert(1,"gender",genders)

    return df_salaries
[df_salaries, df_names] = ReadData()
df_salaries.head()
test = AddGenderToDF(df_salaries.head(), male_names, female_names)
test
test[test["EmployeeName"] == "GARY JIMENEZ"]["first_name"].tolist()[0]
test[test["EmployeeName"] == "GARY JIMENEZ"]["gender"].tolist()[0]
[df_salaries, df_names] = ReadData()

df_salaries = AddGenderToDF(df_salaries, male_names, female_names)



assert df_salaries[df_salaries["EmployeeName"] == "GARY JIMENEZ"]["first_name"].tolist()[0] == "gary"

assert df_salaries[df_salaries["EmployeeName"] == "GARY JIMENEZ"]["gender"].tolist()[0] == "M"
test = df_salaries.head()

test
#select Male by using gender, and get their average salary by using np.mean()

np.mean(test[test["gender"] == "M"]["Total_Pay"].tolist())
def ComputeAvgSalary(df_salaries):

    """

    This function takes the new salary dataframe with gender and

    first_name columns. It returns the the average salary of male

    and female workers. 

    """

    female_avg_salary = 0

    male_avg_salary = 0

    

    female_avg_salary = np.mean(df_salaries[df_salaries["gender"] == "F"]["Total_Pay"].tolist())

    male_avg_salary = np.mean(df_salaries[df_salaries["gender"] == "M"]["Total_Pay"].tolist())

    

    return [male_avg_salary, female_avg_salary]
#previous df_salaries is overrided, get df_salaries again by using ReadData()

[df_salaries, df_names] = ReadData()
df_salaries = AddGenderToDF(df_salaries, male_names, female_names)

[male_avg_salary, female_avg_salary] = ComputeAvgSalary(df_salaries)

assert round(male_avg_salary, 2) == 97774.57

assert round(female_avg_salary, 2) == 83172.18