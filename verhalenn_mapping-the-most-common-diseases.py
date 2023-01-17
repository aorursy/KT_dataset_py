import pandas as pd

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt



outbreak_df = pd.read_csv('../input/Outbreak_240817.csv')
# Find the most common diseases and put them in an array.

most_common_diseases = outbreak_df.disease.value_counts().head().index

print(most_common_diseases)
common_df = outbreak_df[outbreak_df['disease'].isin(most_common_diseases)]

# Make sure that we only have the most common diseases.

common_df.disease.unique()
m = Basemap(projection='mill')

m.drawcoastlines()

for disease in common_df.disease.unique():

    disease_df = common_df[common_df.disease == disease]

    x, y = m(list(disease_df['longitude']), list(disease_df['latitude']))

    m.scatter(x, y, label=disease)

    

plt.legend()

plt.show()
common_df.groupby('disease').region.value_counts()