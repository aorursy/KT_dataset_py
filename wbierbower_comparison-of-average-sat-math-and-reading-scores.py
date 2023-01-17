# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/scores.csv")
df.columns
math_scores = df['Average Score (SAT Math)']

print("max", math_scores.max())

print("min", math_scores.min())

print("mean", math_scores.mean())

print("std", math_scores.std())

math_scores = math_scores.sort_values()

math_scores.plot.bar(title="Average Math Scores")
reading_scores = df['Average Score (SAT Reading)']

print("max", reading_scores.max())

print("min", reading_scores.min())

print("mean", reading_scores.mean())

print("std", reading_scores.std())

reading_scores = reading_scores.sort_values()

reading_scores.plot.bar(title="Average Reading Scores")