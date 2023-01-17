import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv("../input/lcadata.csv")
df.head()
df.describe()
df['1. Gender?'].hist(bins=5)

df['2. What age group do you fall under?'].hist(bins=5)

df['3. Which School are you registered to?'].hist(bins=5)
df['4. What year of study are you currently in?'].hist(bins=5)

df['5. Have you ever met with your designated Academic Advisor?'].hist(bins=5)

df['6. If No, please select relatable reason(s) for not meeting with your academic advisor (Check all that applies)'].hist(bins=5)

df['7. When faced with Academic concerns, who do you speak with? (Check all that applies)'].hist(bins=5)

df['8. Which Academic issue have you experienced? (Check all that applies)'].hist(bins=5)

df['9. How challenging is it to meet with your Academic Advisor or Staff face-to-face to address your academic concerns?'].hist(bins=5)

df['10. Do you have any experience using an online chat application such as the following to communicate with persons? (check all that applies)'].hist(bins=5)

df['11. How interested would you be in using an online chat application for communicating with your Academic Advisor and/or Staff on matters relating to academic concerns, such as grade forgiveness, module selection, grade queries, etc?'].hist(bins=5)

df['12. How beneficial do you think an online chat application solution can be in providing better accessibility to information an advisement for students?'].hist(bins=5)

df['13. What suggestions/concerns do you have regarding the design of an online chat application for communicating with students when handling their academic concerns?'].hist(bins=10)

