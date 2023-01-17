import pandas as pd
df = pd.read_csv('../input/train.csv')
df.columns
df.info()
df.head()
df.tail()
df.describe()
!ls ../input

# from pandas.plotting import scatter_matrix
# scatter_matrix(X_train, c=y_train);
import seaborn as sns
sns.pairplot(df.drop(['PassengerId'], axis=1).dropna(axis=1), hue='Survived', plot_kws={'alpha':0.3});
sns.barplot(x='Sex', y='Survived', hue='Pclass', data=df);
sns.swarmplot(x="Sex", y="Fare", hue="Survived", data=df, alpha=0.3);
### Investigate the correlations between variables
sns.heatmap(df.drop('PassengerId', axis=1).corr(), annot=True, cmap="coolwarm", center=0, square=True);
df.info()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# X = df.select_dtypes(["int64", "float64"]).dropna(axis=1).drop(["PassengerId", "Survived"], axis=1).values
X = df.select_dtypes(["int64", "float64"]).fillna(df.median()).drop(["PassengerId", "Survived"], axis=1).values
y = df['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)