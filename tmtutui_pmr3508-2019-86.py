import pandas as pd
trainDF = pd.read_csv('/kaggle/input/adult-pmr3508/train_data.csv', 

        names=[

          "Age", "Workclass", "Samp_weight", "Education", "Education-Num", "Martial Status",

          "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

          "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
trainDF = trainDF.drop(['Id'])

trainDF.head()
import matplotlib as mpl

%matplotlib inline

mpl.rcParams['figure.dpi'] = 100



from matplotlib import pyplot as plt





plt.figure(figsize=(18,5))



fig = plt.bar(sorted(trainDF.Age.unique()), trainDF['Age'].value_counts().sort_index(), alpha=0.8)





plt.xlabel('Age').set_color('black')

plt.xticks(rotation=90)

[i.set_color("black") for i in plt.gca().get_xticklabels()]





plt.ylabel('Frequency').set_color('black')

[i.set_color("black") for i in plt.gca().get_yticklabels()]





plt.title('Age distribution').set_color('black')

plt.margins(x=0, y=None, tight=True)
print(trainDF['Age'].mean()) # n sei oq aconteceu aki

trainDF.groupby(by="Age").describe()
print(trainDF.shape)

clean_tDF = trainDF.dropna()

print(clean_tDF.shape)
fig = plt.bar(clean_tDF.Workclass.unique(), clean_tDF['Workclass'].value_counts(), alpha=0.8)





plt.xlabel('Workclass').set_color('black')

plt.xticks(rotation=90)

[i.set_color("black") for i in plt.gca().get_xticklabels()]





plt.ylabel('Frequency').set_color('black')

[i.set_color("black") for i in plt.gca().get_yticklabels()]





plt.title('Workclass distribution').set_color('black')

plt.margins(x=0, y=None, tight=True)
fig = plt.figure(figsize=(25,20))



for ed_num,tmpdf in clean_tDF.groupby(by='Education-Num'):

    plt.scatter(sorted(tmpdf['Age']),sorted(tmpdf['Hours per week']),label=ed_num)



plt.legend()

plt.title('Age x Hours per Week')

plt.xlabel('Age')

plt.ylabel('Hours per week')
clean_tDF.Target.value_counts(normalize=True).plot(kind="bar")
X = clean_tDF[["Age", "Education-Num", "Capital Gain", "Capital Loss" ,"Hours per week"]]

Y = clean_tDF.Target
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



best_meanScore = 0



#for n in range(1,20):

#  neigh = KNeighborsClassifier(n_neighbors=n)

#  neigh.fit(X, Y) 

#  for folds in range(2,10):

#    score_rf = cross_val_score(neigh, X, Y, cv=folds, scoring='accuracy').mean()

#    if score_rf > best_meanScore:

#      best_meanScore = score_rf

#      best_pair = [n, folds] #best pair of n and cv

      

#print(best_pair)

#best_meanScore
neigh = KNeighborsClassifier(n_neighbors=14)

neigh.fit(X, Y) 

score_rf = cross_val_score(neigh, X, Y, cv=9, scoring='accuracy')

score_rf
testDF =  pd.read_csv('/kaggle/input/adult-pmr3508/test_data.csv',

        names=[

          "Age", "Workclass", "Samp_weight", "Education", "Education-Num", "Martial Status",

          "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

          "Hours per week", "Country"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
testDF = testDF.drop(["Id"])
testDF.head()
Xtest = testDF[["Age", "Education-Num", "Capital Gain", "Capital Loss" ,"Hours per week"]]
Ypred = neigh.predict(Xtest)
savepath = "predictions.csv" 

prev = pd.DataFrame(Ypred, columns = ["income"]) 

prev.to_csv(savepath, index_label="Id") 

prev
prev.income.value_counts(normalize=True).plot(kind="bar")