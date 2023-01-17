import pandas as pd #For managing data
import seaborn as sns # For plotting

data_train = pd.read_csv('../input/titanic/train.csv')
data_test = pd.read_csv('../input/titanic/test.csv')

data_train.sample(3)

#sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train,
#palette={"male":"green", "female":"blue"})
#sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train,
             #markers=["o","*"], linestyles=["--","--"])

print(data_train.columns.values)