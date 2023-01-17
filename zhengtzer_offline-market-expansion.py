import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



np.random.seed(8)



# Matplotlib and seaborn for plotting

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

matplotlib.rcParams['font.size'] = 24

matplotlib.rcParams['figure.figsize'] = (16, 9)

plt.style.use('ggplot')



import seaborn as sns



from sklearn.cluster import KMeans



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("../input/brazilian-ecommerce/olist_geolocation_dataset.csv", usecols=['geolocation_lat','geolocation_lng'])

df = df.dropna()



df.head()
ax = sns.scatterplot(x="geolocation_lng", y="geolocation_lat", data=df)
df = df.loc[(df['geolocation_lng']<-33) & (df['geolocation_lat']<10)]

plt.scatter(df['geolocation_lng'], df['geolocation_lat'], s=7)
wcss = []



for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(df)

    wcss.append (kmeans.inertia_)



plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
Kmean = KMeans(n_clusters=8)

Kmean.fit(df)
Kmean.cluster_centers_
c1=(-43.11205926,-21.39876995)

c2=(-51.15655913,-28.66327571)

c3=(-37.73534706, -9.49731068)

c4=(-48.142545  ,-16.53194041)

c5=(-59.43186327,-10.53932311)

c6=(-46.74669883,-23.2673302)

c7=(-50.90155881,-23.1401253)

c8=(-46.59041247,-3.53121212)
def calculate_distance(centroid, X, Y):

    distances = []

        

    # Unpack the x and y coordinates of the centroid

    c_x, c_y = centroid

        

    # Iterate over the data points and calculate the distance using the           # given formula

    for x, y in list(zip(X, Y)):

        root_diff_x = (x - c_x) ** 2

        root_diff_y = (y - c_y) ** 2

        distance = np.sqrt(root_diff_x + root_diff_y)

        distances.append(distance)

        

    return distances
df['C1_Distance'] = calculate_distance(c1, df['geolocation_lng'], df['geolocation_lat'])

df['C2_Distance'] = calculate_distance(c2, df['geolocation_lng'], df['geolocation_lat'])

df['C3_Distance'] = calculate_distance(c3, df['geolocation_lng'], df['geolocation_lat'])

df['C4_Distance'] = calculate_distance(c4, df['geolocation_lng'], df['geolocation_lat'])

df['C5_Distance'] = calculate_distance(c5, df['geolocation_lng'], df['geolocation_lat'])

df['C6_Distance'] = calculate_distance(c6, df['geolocation_lng'], df['geolocation_lat'])

df['C7_Distance'] = calculate_distance(c7, df['geolocation_lng'], df['geolocation_lat'])

df['C8_Distance'] = calculate_distance(c8, df['geolocation_lng'], df['geolocation_lat'])
df.head()
df['Cluster'] = df[['C1_Distance', 'C2_Distance', 'C3_Distance', 'C4_Distance', 'C5_Distance', 'C6_Distance','C7_Distance','C8_Distance']].idxmin(axis = 1)

df.head()
df['Cluster'] = df['Cluster'].map({'C1_Distance': 'C1', 'C2_Distance': 'C2', 'C3_Distance': 'C3', 'C4_Distance': 'C4', 'C5_Distance': 'C5','C6_Distance': 'C6','C7_Distance': 'C7','C8_Distance': 'C8'})

df.head()
df['Cluster'].unique()
#plt.yticks(np.arange(-40, 10, 5))

#plt.xticks(np.arange(-75, -30, 5))



plt.scatter(df['geolocation_lng'], df['geolocation_lat'], c=df['Cluster'], s=5)

plt.scatter(c1[0], c1[1], marker='*', s=100, c='yellow')

plt.scatter(c2[0], c2[1], marker='*', s=100, c='grey')

plt.scatter(c3[0], c3[1], marker='*', s=100, c='green')

plt.scatter(c4[0], c4[1], marker='*', s=100, c='blue')

plt.scatter(c5[0], c5[1], marker='*', s=100, c='pink')

plt.scatter(c6[0], c6[1], marker='*', s=100, c='black')

plt.scatter(c7[0], c7[1], marker='*', s=100, c='white')

plt.scatter(c8[0], c8[1], marker='*', s=100, c='purple')