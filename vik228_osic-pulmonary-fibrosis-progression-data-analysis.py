!pip install git+https://github.com/vik228/deeplearning_ai
!pip install nb_black
%load_ext nb_black

import numpy as np

import pandas as pd

import math

from data_visualisation.plot import Plot

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import os
train_data = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

test_data = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")

image_dir = "../input/osic-pulmonary-fibrosis-progression"
train_data.head()
def count_plot(x_key, y_key, df, **kwargs):

    xlabel = kwargs.get("xlabel")

    ylabel = kwargs.get("ylabel")

    title = kwargs.get("title")

    p1 = Plot(df)

    plots = [

        {

            "type": "bar",

            "params": (

                x_key,

                y_key,

                {

                    "axes_settings": {

                        "xlabel": xlabel,

                        "ylabel": ylabel,

                        "title": title,

                    },

                    "ci": None,

                },

            ),

        }

    ]

    p1.plot_bulk(plots, single_figure=True, single_axes=True, figsize=(20, 10))
# The unique number of patients in train and test set

all_patients_train = train_data["Patient"].unique()

print("Number of Unique Patients in train dataset are ", len(all_patients_train))



all_patients_test = test_data["Patient"].unique()

print("Number of Unique Patients in test dataset are ", len(all_patients_test))

# The number of distinct week values

def explore_weeks_attribute(df):

    distinct_values_for_week = df["Weeks"].unique()

    print(

        "Total distinct value for weeks ",

        len(distinct_values_for_week),

    )

    df["count"] = 1

    weeks_df = (

        df.groupby("Weeks")["count"]

        .sum()

        .reset_index()

        .sort_values(by=["count"], ascending=False)

    )

    print("Top 10 week numbers with most entries in data\n", weeks_df.head(10))

    count_plot("Weeks", 

               "count", 

               weeks_df, xlabel="Week numbers", 

               ylabel="count", 

               title="Distribution of week numbers")

print("\nWeek exploration for train data are\n")

explore_weeks_attribute(train_data)

print("\nWeek exploration for test data are\n")

explore_weeks_attribute(test_data)



# Age Variablity

def explore_age_attribute(df):

    df["count"] = 1

    age_values = df["Age"].unique()

    print("Number of unique age values are ", len(age_values))

    groupped_by_age = (

        df.groupby("Age")["count"]

        .sum()

        .reset_index()

        .sort_values(by=["count"], ascending=False)

    )

    print("Top 10 Age numbers with most entries in data\n", groupped_by_age.head(10))

    count_plot(

        "Age",

        "count",

        groupped_by_age,

        xlabel="Age",

        ylabel="count",

        title="Distribution of Age",

    )





print("Age exploration for train data\n")

explore_age_attribute(train_data)

print("Min age in train data is \n", train_data["Age"].min())

print("Max age in train data is \n", train_data["Age"].max())

print("Median age in train data is \n", train_data["Age"].median())

print("Mean age in train data is \n", train_data["Age"].mean())

print("\nAge exploration for test data\n")

explore_age_attribute(test_data)

print("Min age in test data is \n", test_data["Age"].min())

print("Max age in test data is \n", test_data["Age"].max())
# Explore sex attributes





def explore_sex_attribute(df):

    df["count"] = 1

    age_values = df["Sex"].unique()

    print("\nNumber of unique sex values are ", len(age_values))

    groupped_by_sex = (

        df.groupby("Sex")["count"]

        .sum()

        .reset_index()

        .sort_values(by=["count"], ascending=False)

    )

    print(

        "\nNumber of entries in different gender values are \n",

        groupped_by_sex.head(10),

    )

    count_plot(

        "Sex",

        "count",

        groupped_by_sex,

        xlabel="Sex",

        ylabel="Count",

        title="Distribution of Sex",

    )





explore_sex_attribute(train_data)

explore_sex_attribute(test_data)
# Explore SmokingStatus Attribute

def explore_sex_attribute(df):

    df["count"] = 1

    smoking_status_df = df["SmokingStatus"].unique()

    print("\nNumber of unique sex values are ", len(smoking_status_df))

    groupped_by_smoking_status = (

        df.groupby("SmokingStatus")["count"]

        .sum()

        .reset_index()

        .sort_values(by=["count"], ascending=False)

    )

    print(

        "\nNumber of entries in different smoking statuses are \n",

        groupped_by_smoking_status.head(10),

    )

    count_plot(

        "SmokingStatus",

        "count",

        groupped_by_smoking_status,

        xlabel="SmokingStatus",

        ylabel="Count",

        title="Distribution of Smoking Status",

    )





explore_sex_attribute(train_data)

explore_sex_attribute(test_data)
female_df = train_data[train_data["Sex"] == "Female"]

explore_age_attribute(train_data)

explore_age_attribute(female_df)
import pydicom



user_images = os.listdir(image_dir + "/train/ID00007637202177411956430")

nrows = 4

ncols = 4

pic_index = 10

pic_index += 8

user_images_paths = [

    image_dir + "/train/ID00007637202177411956430/" + fname

    for fname in user_images[pic_index - 8 : pic_index]

]

fig = plt.gcf()

fig.set_size_inches(ncols * 4, nrows * 4)

for i, file in enumerate(user_images_paths):

    sp = plt.subplot(nrows, ncols, i + 1)

    sp.axis("Off")  # Don't show axes (or gridlines)

    img = pydicom.dcmread(file)

    plt.imshow(img.pixel_array, cmap=plt.cm.bone)

plt.show()
p1 = Plot(train_data)

p1.distplot(

    "FVC", axes_settings={"title": "Distribution of Percent"}, figure_size=(20, 10)

)
p1.distplot(

    "Percent", axes_settings={"title": "Distribution of Percent"}, figure_size=(20, 10)

)
group_by_week = train_data.groupby("Weeks")["Percent", "FVC",].mean()

group_by_week["Weeks"] = group_by_week.index

group_by_week.head()



p4 = Plot(group_by_week)

regplots_weeks = [

    [

        {

            "type": "regplot",

            "params": (

                "Weeks",

                "FVC",

                {

                    "axes_settings": {"title": "Distribution of FVC by week"},

                    "truncate": False,

                    "color": "c",

                    "order": 3,

                },

            ),

        },

        {

            "type": "regplot",

            "params": (

                "Weeks",

                "Percent",

                {

                    "axes_settings": {"title": "Distribution of Percent by week"},

                    "truncate": False,

                    "color": "m",

                    "order": 3,

                },

            ),

        },

    ]

]

p4.plot_bulk(regplots_weeks, figsize=(25, 10))

group_by_week.head()
group_by_age = train_data.groupby("Age")["Percent", "FVC"].mean()

group_by_age["Age"] = group_by_age.index

group_by_age.head()



p5 = Plot(group_by_age)

regplots_age = [

    [

        {

            "type": "regplot",

            "params": (

                "Age",

                "FVC",

                {

                    "axes_settings": {"title": "Distribution of FVC by Age"},

                    "truncate": False,

                    "order": 2,

                },

            ),

        },

        {

            "type": "regplot",

            "params": (

                "Age",

                "Percent",

                {

                    "axes_settings": {"title": "Distribution of Percent by Age"},

                    "truncate": False,

                    "order": 2,

                },

            ),

        },

    ]

]

p5.plot_bulk(regplots_age, figsize=(25, 10))

group_by_age.head()
group_by_smokingstatus = train_data.groupby("SmokingStatus")["Percent", "FVC"].mean()

group_by_smokingstatus["SmokingStatus"] = group_by_smokingstatus.index

p6 = Plot(group_by_smokingstatus)

regplots_smokingstatus = [

    [

        {

            "type": "bar",

            "params": (

                "SmokingStatus",

                "FVC",

                {"axes_settings": {"title": "Distribution of FVC by SmokingStatus"}},

            ),

        },

        {

            "type": "bar",

            "params": (

                "SmokingStatus",

                "Percent",

                {

                    "axes_settings": {

                        "title": "Distribution of Percent by SmokingStatus"

                    }

                },

            ),

        },

    ]

]

p6.plot_bulk(regplots_smokingstatus, figsize=(25, 10))

group_by_smokingstatus.head()