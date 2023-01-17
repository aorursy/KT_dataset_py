import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

sns.set_style("darkgrid")
%matplotlib inline
from sklearn.datasets import load_iris

data = load_iris()
cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

feature_set = pd.DataFrame(data.data, columns=cols)

species = pd.Series(data.target, name="species").map({
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica'
})


iris_data = pd.merge(feature_set, species, left_index=True, right_index=True)
# iris_data = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
# iris_data.head()
# print a concise summary of a DataFrame.

iris_data.info()
# Generate descriptive statistics.
# Descriptive statistics include those that summarize the central tendency, 
# dispersion and shape of a dataset’s distribution, excluding NaN values.

iris_data.describe()
# check class distribution

_ = sns.countplot(iris_data.species)
_ = plt.title("Class distribution => Balanced Dataset", fontsize=14)
iris_setosa = iris_data.loc[iris_data.species == "Iris-setosa"]
iris_versicolor = iris_data.loc[iris_data.species == "Iris-versicolor"]
iris_virginica = iris_data.loc[iris_data.species == "Iris-virginica"]
print("Setosa mean: ", np.mean(iris_setosa['petal_length']))
print("Setosa corrupted mean: ", np.mean(np.append(iris_setosa['petal_length'], 50)))
print("versicolor mean: ", np.mean(iris_versicolor['petal_length']))
print("virginica mean: ",  np.mean(iris_virginica['petal_length']))
print()
print("Setosa variance: ", np.var(iris_setosa['petal_length']))
print("Setosa corrupted variance: ", np.var(np.append(iris_setosa['petal_length'], 50)))
print("versicolor variance: ", np.var(iris_versicolor['petal_length']))
print("virginica variance: ", np.var(iris_virginica['petal_length']))
print() 
print("Setosa std: ", np.std(iris_setosa['petal_length']))
print("Setosa corrupted std: ", np.std(np.append(iris_setosa['petal_length'], 50)))
print("versicolor std: ", np.std(iris_versicolor['petal_length']))
print("virginica std: ", np.std(iris_virginica['petal_length']))
print("Median: ")
print("Setosa Median: ", np.median(iris_setosa['petal_length']))
print("Setosa corrupted Median: ", np.median(np.append(iris_setosa['petal_length'], 50)))
print("versicolor Median: ", np.median(iris_versicolor['petal_length']))
print("virginica Median: ",  np.median(iris_virginica['petal_length']))

print("\nQuantiles: [0, 25, 50, 75]")
print("Setosa Quantile: ", np.percentile(iris_setosa['petal_length'], np.arange(0, 100, 25)))
print("Setosa corrupted Quantile: ", np.percentile(np.append(iris_setosa['petal_length'], 50), np.arange(0, 100, 25)))
print("versicolor Quantile: ", np.percentile(iris_versicolor['petal_length'], np.arange(0, 100, 25)))
print("virginica Quantile: ", np.percentile(iris_virginica['petal_length'], np.arange(0, 100, 25)))

print("\n90th Percentiles")
print("Setosa Percentile: ", np.percentile(iris_setosa['petal_length'], 90))
print("Setosa corrupted Percentile: ", np.percentile(np.append(iris_setosa['petal_length'], 50), 90))
print("versicolor Percentile: ", np.percentile(iris_versicolor['petal_length'], 90))
print("virginica Percentile: ", np.percentile(iris_virginica['petal_length'], 90))

from statsmodels import robust
print("\nMedian Absolute Deviation")
print("Setosa MAD: ", robust.mad(iris_setosa['petal_length']))
print("Setosa corrupted MAD: ", robust.mad(np.append(iris_setosa['petal_length'], 50)))
print("versicolor MAD: ", robust.mad(iris_versicolor['petal_length']))
print("virginica MAD: ", robust.mad(iris_virginica['petal_length']))
fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for no, column in enumerate(iris_data.columns[:-1], 1):
    ax = fig.add_subplot(2, 2, no)
    sns.distplot(iris_data.loc[iris_data.species == 'Iris-setosa', f"{column}"], label="Setosa")
    sns.distplot(iris_data.loc[iris_data.species == 'Iris-versicolor', f"{column}"], label="Versicolor")
    sns.distplot(iris_data.loc[iris_data.species == 'Iris-virginica', f"{column}"], label="Virginica")
    ax.legend()

plt.tight_layout(pad=2.0)
plt.show()
counts, bin_edges = np.histogram(iris_data.loc[iris_data.species == 'Iris-setosa', 'petal_length'],
                                bins=10, density=True)

pdf = counts/sum(counts)
print("PDF: ", pdf)
print("CDF: ", bin_edges)

# cdf 
cdf = np.cumsum(pdf)
_ = plt.plot(bin_edges[1:], pdf)
_ = plt.plot(bin_edges[1:], cdf)
fig = plt.figure(figsize=(10, 6))

for i, cls in enumerate(iris_data.species.unique(), 1):
    counts, bin_edges = np.histogram(iris_data.loc[iris_data.species == f'{cls}', 'petal_length'],
                                    bins=10, density=True)

    pdf = counts/sum(counts)
    # print("PDF: ", pdf)
    # print("CDF: ", bin_edges)

    # cdf 
    cdf = np.cumsum(pdf)
    _ = plt.plot(bin_edges[1:], pdf)
    _ = plt.plot(bin_edges[1:], cdf, label=f'{cls}')

plt.title(f"{cls}: PDF & CDF plot")
plt.xlabel("petal_length")
plt.legend()
plt.show()
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(9, 40))
outer = gridspec.GridSpec(4, 1, wspace=0.2, hspace=0.2)

for i, col in enumerate(iris_data.columns[:-1]):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                    subplot_spec=outer[i], wspace=0.2, hspace=0.4)

    ax = plt.Subplot(fig, inner[0])
    _ = sns.boxplot(y="species", x=f"{col}", data=iris_data, ax=ax)
    _ = sns.stripplot(y="species", x=f"{col}", data=iris_data,  jitter=True, dodge=True, linewidth=1, ax=ax)
    _ = ax.set_title("Box Plot")
    fig.add_subplot(ax)

    ax = plt.Subplot(fig, inner[1])
    _ = sns.violinplot(y="species", x=f"{col}", data=iris_data, inner='quartile', ax=ax)
    # _ = sns.stripplot(x="species", y="petal_length", data=iris_data, jitter=True, dodge=True, linewidth=1, ax=ax)
    _ = ax.set_title("Violin Plot")
    fig.add_subplot(ax)
fig.show()
# experimenting

def simple_rule(subset):
    cls = []
    for idx, row in subset.iterrows():
        if row['petal_length'] <= 2:
            cls.append("Iris-setosa")
        elif row['petal_length'] > 2 and row['petal_length'] <=4.6:
            cls.append("Iris-versicolor")
        else:
            cls.append("Iris-virginica")
    # accuracy
    cls = np.array(cls)
        
    return accuracy_score(cls, subset.species.values)
iris_data.sample(frac=1)

random_idx = np.random.choice(range(0, 150), 20)
simple_rule(iris_data.sample(frac=1).iloc[random_idx])
plt.figure(figsize=(8, 6))
_ = sns.heatmap(iris_data.corr(), vmin=-1, vmax=1, annot=True, cmap='afmhot')
_ = sns.relplot(x='petal_length', y='petal_width', hue='species', data=iris_data, height=7)
_ = plt.title("Scatter plot", fontsize=14)
# From the scatterplot we can clearly see that Iris-setosa can be easily identified/linearly seperated using sepal_length and sepal_width
# whereas Iris-versicolor and Iris-virginia have almost the same distribution and rather difficult to seperate


_ = plt.figure(figsize=(15, 10))
_ = sns.pairplot(iris_data, hue="species", height=3, diag_kind="kde")
g = sns.jointplot(x="petal_length", y="petal_width", data=iris_setosa, kind="kde")
g.plot_joint(plt.scatter, c="k", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$Petal$ $length$", "$Petal$ width$");
import plotly.express as px
df = px.data.iris()
fig = px.scatter_3d(iris_data, x='petal_length', y='petal_width', z='sepal_length',
                    color='species')
fig.show()
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import PCA
# Seperating X and y
X = iris_data.drop(['species'], axis=1)
y = iris_data['species'].map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
})
print(X.shape)
print(y.shape)
models = []

models.append(("LogisticRegression", LogisticRegression(max_iter=1000)))
models.append(("SVC", SVC(kernel="rbf", gamma=5, C=0.001, max_iter=1000)))

models.append(("KNeighbors", KNeighborsClassifier(n_neighbors=12)))
models.append(("DecisionTree", DecisionTreeClassifier()))
models.append(("RandomForest", RandomForestClassifier()))
rf2 = RandomForestClassifier(n_estimators=100, criterion='gini',
                                max_depth=10, random_state=42, max_features=None)
models.append(("RandomForest2", rf2))
models.append(('NB', GaussianNB()))
models.append(("MLPClassifier", MLPClassifier(hidden_layer_sizes=(10, 10), solver='adam', max_iter=2000, learning_rate='adaptive', random_state=42)))
# naive feature selection

for i in range(1, 5):
    cols = X.columns[:i]
    X_temp = X[cols].values
    results = []
    names = []
    for name, model in models:
        try:
            result = cross_val_score(model, X[cols], y, cv=5, scoring='accuracy')
        except:
            result = cross_val_score(model, X[cols].reshape(-1, 1), y, cv=5, scoring='accuracy')
        
        names.append(name)
        results.append(result)
    
    print(f"Using features: {cols}")
    
    for i in range(len(names)):
        # f"{'1':0>8}
        print(f"Algo: {names[i]}, Result: {round(results[i].mean(), 2)}")
    print()
single_feature_models = models[:]
single_feature_models.pop(2)
single_feature_models.insert(2, ("KNeighbors", KNeighborsClassifier(n_neighbors=3)))

two_feature_models = models[:]
two_feature_models.pop(2)
two_feature_models.insert(2, ("KNeighbors", KNeighborsClassifier(n_neighbors=5)))

X_selected_1 = X[['petal_length']].values
X_selected_2 = X[['petal_length', 'petal_width']].values

X_ = [X_selected_1, X_selected_2]
y = y.ravel()

mods = [single_feature_models, two_feature_models]
for i in range(2):   
    curr_models = mods[i]
    names = []
    results = []
    for name, mod in curr_models:    
        if i == 0:
            result = cross_val_score(mod, X_selected_1.reshape(-1, 1), y, cv=5, scoring='accuracy')        
        else:
            result = cross_val_score(mod, X_selected_2, y, cv=5, scoring='accuracy')            
        
        names.append(name)
        results.append(result) 

    print(f"Features: {X_[i].shape[1]}")
    for j in range(len(names)):
        print(f"Algo: {names[j]}, Result: {round(results[j].mean(), 2)}")
    print()
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
# Seperating X and y
X = iris_data.drop(['species'], axis=1)
y = iris_data['species'].map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
})
print(X.shape)
print(y.shape)
x_train, x_val, y_train, y_val = train_test_split(X, y, shuffle=True, stratify=y, 
                                                  random_state=42, test_size=0.1)
model = Sequential([
                    Input(shape=(4,)),
                    Dense(10, activation='relu'),
                    Dense(10, activation='relu'),
                    Dense(3, activation='softmax'),
])

model.compile(loss='sparse_categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), verbose=0)
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, 101), history.history['loss'], label="Loss")
plt.plot(range(1, 101), history.history['val_loss'], label="validation_loss")
plt.legend()
plt.title("Epoch Vs. loss")
plt.xlabel("Epoch")
plt.ylabel("loss")

plt.subplot(1, 2, 2)
plt.plot(range(1, 101), history.history['accuracy'], label="accuracy")
plt.plot(range(1, 101), history.history['val_accuracy'], label="validation_accuracy")
plt.legend()
plt.title("Epoch Vs. accuracy")
plt.xlabel("Epoch")
plt.ylabel("accuracy")

plt.show()
# Seperating X and y
X = iris_data.drop(['species'], axis=1)
y = iris_data['species'].map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
})
print(X.shape)
print(y.shape)
pca = PCA(n_components=0.95)
X_transformed = pca.fit_transform(X)
print(X_transformed.shape)
print(pca.n_components_)
print(pca.explained_variance_ratio_)
x_train, x_val, y_train, y_val = train_test_split(X_transformed, y, shuffle=True, stratify=y, test_size=0.1)

log = LogisticRegression(max_iter=500)
log.fit(x_train, y_train)
pred = log.predict(x_val)
accuracy_score(y_val, pred)