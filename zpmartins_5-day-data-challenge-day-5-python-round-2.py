# Following

# http://mailchi.mp/422c4b65434f/data-challenge-day-1-read-in-and-summarize-a-csv-file-2576433
import numpy as np 

import scipy.stats

import pandas as pd

import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/anonymous-survey-responses.csv')
dataset.describe().transpose()
scipy.stats.chisquare(

    dataset["Have you ever taken a course in statistics?"].value_counts())
scipy.stats.chisquare(

    dataset["Do you have any previous experience with programming?"].value_counts())
contingencyTable = pd.crosstab(

    dataset["Have you ever taken a course in statistics?"],

    dataset["Do you have any previous experience with programming?"]) 
scipy.stats.chi2_contingency(contingencyTable)


