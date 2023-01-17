import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        import pandas as pd
my_filepath = "../input/coronavirusdataset/Case.csv"
data = pd.read_csv(my_filepath)
data.head(10)
data=data.groupby(['city'])['confirmed','city'].mean()
data.sort_values(by=['confirmed'], ascending=False,inplace=True)

data.reset_index(level=0, inplace=True)
data = data.iloc[1:10]
plt.figure(figsize=(20,10))

sns.barplot(x=data['city'], y=data['confirmed'])