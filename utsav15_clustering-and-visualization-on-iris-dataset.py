# Importing Modules
from sklearn import datasets
import matplotlib.pyplot as plt
%matplotlib inline
iris_df = datasets.load_iris()
# Features
print(iris_df.feature_names)
# Targets
print(iris_df.target)
# Target Names
print(iris_df.target_names)
label = {0: 'red', 1: 'blue', 2: 'green'}
# Dataset Slicing
x_axis = iris_df.data[:, 0]  # Sepal Length
y_axis = iris_df.data[:, 2]  # Sepal Width
# Plotting
plt.scatter(x_axis, y_axis, c=iris_df.target)
plt.show()
from sklearn.cluster import KMeans

# Declaring Model
model = KMeans(n_clusters=3)
# Fitting Model
model.fit(iris_df.data)
# Predicitng a single input
predicted_label = model.predict([[7.2, 3.5, 0.8, 1.6]])


# Prediction on the entire data
all_predictions = model.predict(iris_df.data)

predicted_label
