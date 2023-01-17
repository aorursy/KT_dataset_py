import numpy as np

import pandas as pd 



%matplotlib inline

pd.set_option('display.max_rows', 20)
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
df = pd.read_csv("../input/human_development.csv")
df
df.info()
df["Life Expectancy at Birth"].plot.hist(bins=11, grid=False, rwidth=0.95); 
df["uus"] = df["Gross National Income (GNI) per Capita"].str.replace(",", "")

df["uus"] = pd.to_numeric(df["uus"])
df.groupby(["Expected Years of Education", "uus"]).count()
df = df.sort_values("uus", ascending = False)

pd.DataFrame({"Riik" : df["Country"],

             "SKP" : df["uus"]})
df.plot.scatter("Expected Years of Education", "uus", alpha=0.5);