import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# see what files are available at the ../input directory
# read USvideos.csv   ( csv - comma separated values)
youtube_USvideos = pd.read_csv("../input/USvideos.csv")
youtube_USvideos.head()  # head will printout a tabular data for the first 5 rows
print(youtube_USvideos.columns) # what are the columns in the dataframe
youtube_USvideos.tail()   # what's there at the end?
print(youtube_USvideos.describe()) # summary 
youtube_USvideos.dtypes  # what the types
#lets see if we can infer the types
youtube_USvideos.infer_objects().dtypes  

# import matplotlib for visualization
import matplotlib.pyplot as plt

# videos with more than 100 million views
youtube_USvideos[youtube_USvideos.views > 100000000]['views'].count()
# horizontal bar plot with x-axis being "title" & y-axis being "views"
youtube_USvideos[youtube_USvideos.views > 100000000].plot(figsize=(20,20), kind="barh", rot=15, x="title", y="views")