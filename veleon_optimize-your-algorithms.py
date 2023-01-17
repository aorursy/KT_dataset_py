import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings



warnings.filterwarnings("ignore")
data = pd.read_csv('../input/data.csv')
data.head(5)
data.describe()
data.drop('Unnamed: 32', axis = 1, inplace = True)
print('Data has {} missing values.'.format(data.isnull().sum().sum()))
sns.countplot(data['diagnosis'])
data.groupby(['diagnosis']).mean()[['radius_mean', 'texture_mean', ]].plot.barh()
data.groupby(['diagnosis']).mean()['perimeter_mean'].plot.barh()
data.groupby(['diagnosis']).mean()[['smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'fractal_dimension_mean' ]].plot.barh(figsize = (8,4))
data.groupby(['diagnosis']).mean()[['fractal_dimension_mean', 'fractal_dimension_worst',  ]].plot.barh(figsize = (8,4))
data.groupby(['diagnosis']).mean()['fractal_dimension_se'].plot.barh(figsize = (8,4))

data['diagnosis'].replace(to_replace='M', value = 1, inplace=True)
data['diagnosis'].replace(to_replace='B', value = 0, inplace=True)
sns.heatmap(data[['diagnosis','radius_mean', 'texture_mean', 'perimeter_mean', 'smoothness_mean']].corr(), annot=True)
sns.heatmap(data[['diagnosis','compactness_mean', 'concavity_mean', 'concave points_mean', 'fractal_dimension_mean']].corr(), annot=True)
sns.heatmap(data[['diagnosis', 'fractal_dimension_mean', 'fractal_dimension_worst', 'fractal_dimension_se']].corr(), annot=True)
y = data['diagnosis']
X = data.drop(['diagnosis', 'id'], axis=1)
from sklearn import ensemble, linear_model, svm, neighbors, gaussian_process, naive_bayes, tree 

scoreFrame = pd.DataFrame(columns = ['Algorithm Name', 'Average', 'Standard Deviation'])

algList=[
    #linear
    linear_model.Ridge(random_state=0),
    linear_model.SGDClassifier(random_state=0),
    #Neighbors
    neighbors.KNeighborsClassifier(),
    #SVM
    svm.SVC(),
    #Gaussian Process
    gaussian_process.GaussianProcessClassifier(random_state=0),
    #Naive Bayes
    naive_bayes.GaussianNB(),
    #Tree
    tree.DecisionTreeClassifier(random_state=0),
    #Ensemble
    ensemble.GradientBoostingClassifier(random_state=0),
    ensemble.RandomForestClassifier(random_state=0),
    ensemble.ExtraTreesClassifier(random_state=0),
    ensemble.AdaBoostClassifier(random_state=0)
]
from sklearn.model_selection import cross_val_score

for alg in algList:
    scores = cross_val_score(alg, X, y, cv = 10)
    algName = alg.__class__.__name__
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    scoreFrame.loc[len(scoreFrame)] = [algName, scoreAverage, scoreSTD]
scoreFrame.sort_values('Average', ascending=False)
svmPenaltyFrame = pd.DataFrame(columns = ['C', 'Average', 'Standard Deviation'])

for c in [0.00001, 0.0001, 0.001, 0.01, 0.1,1,10,100,1000]:
    alg = svm.SVC(C=c)
    scores = cross_val_score(alg, X, y, cv = 3)
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    svmPenaltyFrame.loc[len(svmPenaltyFrame)] = [c, scoreAverage, scoreSTD]
  
svmPenaltyFrame.sort_values('Average', ascending=False).head(10)
svmGammaFrame = pd.DataFrame(columns = ['Gamma', 'Average', 'Standard Deviation'])

for g in [0.001, 0.01, 0.1,1,10,100,1000]:
    alg = svm.SVC(gamma=g, C=0.1)
    scores = cross_val_score(alg, X, y, cv = 3)
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    svmGammaFrame.loc[len(svmGammaFrame)] = [g, scoreAverage, scoreSTD]
  
svmGammaFrame.sort_values('Average', ascending=False).head(10)
for g in range(1,1000):
    g = g/1000000
    alg = svm.SVC(gamma=g, C=0.1)
    scores = cross_val_score(alg, X, y, cv = 3)
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    svmGammaFrame.loc[len(svmGammaFrame)] = [g, scoreAverage, scoreSTD]
   
svmGammaFrame.sort_values('Average', ascending=False).head(10)
optimalSVMGamma = svmGammaFrame.sort_values('Average', ascending=False).iloc[0].values[0]
svmKernelFrame = pd.DataFrame(columns = ['Kernel', 'Average', 'Standard Deviation'])
kernelList = ['linear', 'poly', 'rbf']
for k in kernelList:
    alg = svm.SVC(gamma=optimalSVMGamma, kernel=k, C=0.1)
    scores = cross_val_score(alg, X, y, cv = 3)
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    svmKernelFrame.loc[len(svmKernelFrame)] = [k, scoreAverage, scoreSTD]
    
svmKernelFrame.sort_values('Average', ascending=False).head(10)
optimalSVMKernel = 'poly'
svmDegreeFrame = pd.DataFrame(columns = ['Degrees', 'Average', 'Standard Deviation'])

for d in range(1,4):
    alg = svm.SVC(gamma=optimalSVMGamma, kernel='poly', degree=d, C=0.1)
    scores = cross_val_score(alg, X, y, cv = 3)
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    svmDegreeFrame.loc[len(svmDegreeFrame)] = [d, scoreAverage, scoreSTD]
   
svmDegreeFrame.sort_values('Average', ascending=False).head(10)
optimalSVMDegree = svmDegreeFrame.sort_values('Average', ascending=False).iloc[0].values[0]
svmPenaltyFrame = pd.DataFrame(columns = ['C', 'Average', 'Standard Deviation'])

for c in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]:
    alg = svm.SVC(gamma=optimalSVMGamma, kernel=optimalSVMKernel, degree=optimalSVMDegree, C=c)
    scores = cross_val_score(alg, X, y, cv = 3)
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    svmPenaltyFrame.loc[len(svmPenaltyFrame)] = [c, scoreAverage, scoreSTD]
  
svmPenaltyFrame.sort_values('Average', ascending=False).head(10)
optimalSVMPenalty = svmPenaltyFrame.sort_values('Average', ascending=False).iloc[0].values[0]
ridgeFrame = pd.DataFrame(columns = ['Alpha', 'Average', 'Standard Deviation'])

for a in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    alg = linear_model.Ridge(alpha = a)
    scores = cross_val_score(alg, X, y, cv = 3)
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    ridgeFrame.loc[len(ridgeFrame)] = [a, scoreAverage, scoreSTD]

ridgeFrame.sort_values('Average', ascending=False).head(10)
ridgeFrame = pd.DataFrame(columns = ['Alpha', 'Average', 'Standard Deviation'])

for a in range(1,1000):
    a = a/100000
    alg = linear_model.Ridge(alpha = a)
    scores = cross_val_score(alg, X, y, cv = 3)
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    ridgeFrame.loc[len(ridgeFrame)] = [a, scoreAverage, scoreSTD]

ridgeFrame.sort_values('Average', ascending=False).head(10)
optimalRidgeAlpha = ridgeFrame.sort_values('Average', ascending=False).iloc[0].values[0]
sns.relplot(x = 'Alpha', y = 'Average', data=ridgeFrame, kind="line")
for a in [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]:
    alg = linear_model.Ridge(alpha = a)
    scores = cross_val_score(alg, X, y, cv = 3)
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    ridgeFrame.loc[len(ridgeFrame)] = [a, scoreAverage, scoreSTD]

sns.relplot(x = 'Alpha', y = 'Average', data=ridgeFrame, kind="line")
treeSampleFrame = pd.DataFrame(columns = ['Samples', 'Average', 'Standard Deviation'])

for n in range(1,20):
    alg = tree.DecisionTreeClassifier(min_samples_leaf = n, random_state=0)
    scores = cross_val_score(alg, X, y, cv = 5)
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    treeSampleFrame.loc[len(treeSampleFrame)] = [n, scoreAverage, scoreSTD]

treeSampleFrame.sort_values('Average', ascending=False).head(5)
optimalTreeSamples = int(treeSampleFrame.sort_values('Average', ascending=False).iloc[0].values[0])
sns.relplot(x = 'Samples', y = 'Average', data=treeSampleFrame, kind="line")
treeDepthFrame = pd.DataFrame(columns = ['Depth', 'Average', 'Standard Deviation'])

for d in range(1,20):
    alg = tree.DecisionTreeClassifier(min_samples_leaf = optimalTreeSamples, max_depth = d, random_state=0)
    scores = cross_val_score(alg, X, y, cv = 5)
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    treeDepthFrame.loc[len(treeDepthFrame)] = [d, scoreAverage, scoreSTD]

treeDepthFrame.sort_values('Average', ascending=False)
optimalTreeDepth = int(treeDepthFrame.sort_values('Average', ascending=False).iloc[0].values[0])
sns.relplot(x = 'Depth', y = 'Average', data=treeDepthFrame, kind="line")
treeFrame = pd.DataFrame(columns = ['Depth', 'Samples', 'Average', 'Standard Deviation'])

for n in range(1,20):
    for d in range(1, 10):
        alg = tree.DecisionTreeClassifier(min_samples_leaf = n, max_depth=d, random_state=0)
        scores = cross_val_score(alg, X, y, cv = 5)
        scoreAverage = scores.mean()
        scoreSTD = scores.std() * 2
        treeFrame.loc[len(treeFrame)] = [d, n, scoreAverage, scoreSTD]

treeFrame.sort_values('Average', ascending=False).head(10)
optimalTreeDepth = int(treeFrame.sort_values('Average', ascending=False).iloc[0].values[0])
optimalTreeSamples = int(treeFrame.sort_values('Average', ascending=False).iloc[0].values[1])
sns.heatmap(treeFrame[['Average', 'Depth', 'Samples']].corr(), annot=True)
DTC = tree.DecisionTreeClassifier(min_samples_leaf = optimalTreeSamples, max_depth=optimalTreeDepth, random_state=0)
DTC.fit(X,y)
import graphviz
dot_data = tree.export_graphviz(DTC, feature_names=X.columns.values, class_names=['B', 'M'], filled=True )
graphviz.Source(dot_data) 

kNeighborFrame = pd.DataFrame(columns = ['Neighbors', 'Average', 'Standard Deviation'])
for n in range(1,50):
    alg = neighbors.KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(alg, X, y, cv = 5)
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    kNeighborFrame.loc[len(kNeighborFrame)] = [n, scoreAverage, scoreSTD]
kNeighborFrame.sort_values('Average', ascending=False).head(5)
optimalKNNNeighbors = int(kNeighborFrame.sort_values('Average', ascending=False).iloc[0].values[0])
sns.relplot(x="Neighbors", y='Average', data = kNeighborFrame, kind = "line")
kAlgorithmFrame = pd.DataFrame(columns = ['Algorithm', 'Average', 'Standard Deviation'])
for a in ['ball_tree', 'kd_tree', 'brute','auto']:
    alg = neighbors.KNeighborsClassifier(n_neighbors=optimalKNNNeighbors ,algorithm = a)
    scores = cross_val_score(alg, X, y, cv = 5)
    scoreAverage = scores.mean()
    scoreSTD = scores.std() * 2
    kAlgorithmFrame.loc[len(kAlgorithmFrame)] = [a, scoreAverage, scoreSTD]

kAlgorithmFrame.sort_values('Average', ascending=False).head(10)
optimalKNNAlg = kAlgorithmFrame.sort_values('Average', ascending=False).iloc[0].values[0]
kWeightFrame = pd.DataFrame(columns = ['Weight', 'Average', 'Standard Deviation'])
for w in ['uniform', 'distance']:
        alg = neighbors.KNeighborsClassifier(weights=w, algorithm = optimalKNNAlg, n_neighbors=optimalKNNNeighbors)
        scores = cross_val_score(alg, X, y, cv = 5)
        scoreAverage = scores.mean()
        scoreSTD = scores.std() * 2
        kWeightFrame.loc[len(kWeightFrame)] = [w, scoreAverage, scoreSTD]

kWeightFrame.sort_values('Average', ascending=False).head(10)
optimalKNNWeight = kWeightFrame.sort_values('Average', ascending=False).iloc[0].values[0]
extraTreeFrame = pd.DataFrame(columns = ['n_estimators', 'Depth', 'Samples', 'Average', 'Standard Deviation'])

for n in range(1, 100, 10):
    for d in range(1, 30, 5):
        for s in range(1, 10):
            alg = ensemble.ExtraTreesClassifier(n_estimators=n, max_depth=d, min_samples_leaf=s ,random_state=0)
            scores = cross_val_score(alg, X, y, cv = 5)
            scoreAverage = scores.mean()
            scoreSTD = scores.std() * 2
            extraTreeFrame.loc[len(extraTreeFrame)] = [n, d, s, scoreAverage, scoreSTD]

extraTreeFrame.sort_values('Average', ascending=False).head(5)
sns.heatmap(extraTreeFrame[['Average','n_estimators', 'Depth', 'Samples']].corr(), annot=True)
sns.relplot(x = 'n_estimators', y = 'Average', data=extraTreeFrame, kind="line")
sns.relplot(x = 'Depth', y = 'Average', data=extraTreeFrame, kind="line")
extraTreeFrame = pd.DataFrame(columns = ['n_estimators', 'Depth', 'Average', 'Standard Deviation'])

for n in range(10, 50):
    for d in range(10, 30):
        alg = ensemble.ExtraTreesClassifier(n_estimators=n, max_depth=d, random_state=0)
        scores = cross_val_score(alg, X, y, cv = 5)
        scoreAverage = scores.mean()
        scoreSTD = scores.std() * 2
        extraTreeFrame.loc[len(extraTreeFrame)] = [n, d, scoreAverage, scoreSTD]

extraTreeFrame.sort_values('Average', ascending=False).head(10)
sns.relplot(x = 'n_estimators', y = 'Average', data=extraTreeFrame, kind="line")
sns.relplot(x = 'Depth', y = 'Average', data=extraTreeFrame, kind="line")
optimalExtraTreeN = int(extraTreeFrame.sort_values('Average', ascending=False).iloc[0].values[0])
optimalExtraTreeDepth = int(extraTreeFrame.sort_values('Average', ascending=False).iloc[0].values[1])
randomForestFrame = pd.DataFrame(columns = ['n_estimators', 'Depth', 'Samples', 'Average', 'Standard Deviation'])

for n in range(1, 100, 10):
    for d in range(1, 30, 5):
        for s in range(2, 5):
            alg = ensemble.RandomForestClassifier(n_estimators=n, max_depth=d, min_samples_split=s, random_state=0)
            scores = cross_val_score(alg, X, y, cv = 5)
            scoreAverage = scores.mean()
            scoreSTD = scores.std() * 2
            randomForestFrame.loc[len(randomForestFrame)] = [n, d, s, scoreAverage, scoreSTD]

randomForest = randomForestFrame.sort_values('Average', ascending=False)
randomForest.head(10)
sns.relplot(x = 'n_estimators', y = 'Average', data = randomForestFrame, kind="line")
sns.relplot(x = 'Depth', y = 'Average', data=randomForestFrame, kind="line")
sns.relplot(x = 'Samples', y = 'Average', data=randomForestFrame)
sns.heatmap(randomForestFrame[['Average','n_estimators', 'Depth', 'Samples']].corr(), annot=True)
randomForestFrame = pd.DataFrame(columns = ['n_estimators', 'Depth', 'Samples', 'Average', 'Standard Deviation'])

for n in range(40,80,2):
    for d in range(5,25,5):
        for s in range(2, 5):
            alg = ensemble.RandomForestClassifier(n_estimators=n, max_depth=d, min_samples_split=s, random_state=0)
            scores = cross_val_score(alg, X, y, cv = 5)
            scoreAverage = scores.mean()
            scoreSTD = scores.std() * 2
            randomForestFrame.loc[len(randomForestFrame)] = [n, d, s, scoreAverage, scoreSTD]

randomForestFrame.sort_values('Average', ascending=False).head(10)
optimalRandomForestN = int(randomForestFrame.sort_values('Average', ascending=False).iloc[0].values[0])
optimalRandomForestDepth = int(randomForestFrame.sort_values('Average', ascending=False).iloc[0].values[1])
optimalRandomForestSamples = int(randomForestFrame.sort_values('Average', ascending=False).iloc[0].values[2])
sns.relplot(x = 'n_estimators', y = 'Average', data=randomForestFrame, kind="line")
gradientBoostingTotalFrame = pd.DataFrame(columns = ['n_estimators', 'Depth', 'learning rate', 'Average', 'Standard Deviation'])

for n in range(1, 100, 10):
    for d in range(1, 30, 5):
        for l in range(1, 11):
            l = l/10
            alg = ensemble.GradientBoostingClassifier(n_estimators = n, learning_rate = l, max_depth = d, random_state = 0)
            scores = cross_val_score(alg, X, y, cv = 5)
            scoreAverage = scores.mean()
            scoreSTD = scores.std() * 2
            gradientBoostingTotalFrame.loc[len(gradientBoostingTotalFrame)] = [n, d, l, scoreAverage, scoreSTD]

gradientBoostingTotalFrame.sort_values('Average', ascending=False).head(5)
sns.heatmap(gradientBoostingTotalFrame[['Average','n_estimators', 'Depth', 'learning rate']].corr(), annot=True)
sns.relplot(x = 'n_estimators', y = 'Average', data=gradientBoostingTotalFrame, kind="line")
sns.relplot(x = 'Depth', y = 'Average', data=gradientBoostingTotalFrame, kind="line")
sns.relplot(x = 'learning rate', y = 'Average', data=gradientBoostingTotalFrame, kind="line")
gradientBoostingFrame = pd.DataFrame(columns = ['n_estimators', 'learning rate', 'Average', 'Standard Deviation'])
for n in range(60, 150):
    for l in range(1, 11):
        l = l / 10
        alg = ensemble.GradientBoostingClassifier(n_estimators = n, learning_rate = l, random_state = 0)
        scores = cross_val_score(alg, X, y, cv = 5)
        scoreAverage = scores.mean()
        scoreSTD = scores.std() * 2
        gradientBoostingFrame.loc[len(gradientBoostingFrame)] = [n, l, scoreAverage, scoreSTD]

gradientBoostingFrame.sort_values('Average', ascending=False).head(5)
optimalGBCN = int(gradientBoostingFrame.sort_values('Average', ascending=False).iloc[0].values[0])
optimalGBCLearningRate = gradientBoostingFrame.sort_values('Average', ascending=False).iloc[0].values[1]

optimalGBCDepth = int(gradientBoostingTotalFrame.sort_values('Average', ascending=False).iloc[0].values[1])
adaBoostFrame = pd.DataFrame(columns = ['Name', 'n_estimators', 'Learning Rate', 'Average', 'Standard Deviation'])
for b in [    
    tree.DecisionTreeClassifier(min_samples_leaf = optimalTreeSamples, max_depth= optimalTreeDepth, random_state=0),
    ensemble.ExtraTreesClassifier(n_estimators=optimalExtraTreeN, max_depth=optimalExtraTreeDepth, random_state=0),
    ensemble.RandomForestClassifier(n_estimators=optimalRandomForestN, max_depth=optimalRandomForestDepth, min_samples_split=optimalRandomForestSamples, random_state=0),
    ensemble.GradientBoostingClassifier(n_estimators = optimalGBCN, learning_rate = optimalGBCLearningRate, max_depth = 1, random_state=0),
]:
    for n in range(1,100, 5):
        for l in range(1,11):
            l = l/10
            alg = ensemble.AdaBoostClassifier(n_estimators = n, base_estimator=b, learning_rate = l, random_state = 0)
            scores = cross_val_score(alg, X, y, cv = 5)
            scoreAverage = scores.mean()
            scoreSTD = scores.std() * 2
            adaBoostFrame.loc[len(adaBoostFrame)] = [b.__class__.__name__, n, l, scoreAverage, scoreSTD]

adaBoostFrame.sort_values('Average', ascending=False).head(10)
optimalAdaBoostN = int(adaBoostFrame.sort_values('Average', ascending=False).iloc[0].values[1])
optimalAdaBoostLearningRate = adaBoostFrame.sort_values('Average', ascending=False).iloc[0].values[2]
optimalAdaBoostBase = tree.DecisionTreeClassifier(min_samples_leaf = optimalTreeSamples, max_depth= optimalTreeDepth, random_state=0)
finalScoreFrame = pd.DataFrame(columns = ['Algorithm Name', 'Average', 'Standard Deviation'])
alg = svm.SVC(gamma=optimalSVMGamma, kernel=optimalSVMKernel, degree=optimalSVMDegree, C = optimalSVMPenalty)
scores = cross_val_score(alg, X, y, cv = 5)
scoreAverage = scores.mean()
print('Final SVM Score: {:01.5f}'.format(scoreAverage))
scoreSTD = scores.std() * 2
finalScoreFrame.loc[len(finalScoreFrame)] = [alg.__class__.__name__, scoreAverage, scoreSTD]
alg = linear_model.Ridge(alpha = optimalRidgeAlpha)
scores = cross_val_score(alg, X, y, cv = 5)
scoreAverage = scores.mean()
print('Final Ridge Score: {:01.5f}'.format(scoreAverage))
scoreSTD = scores.std() * 2
finalScoreFrame.loc[len(finalScoreFrame)] = [alg.__class__.__name__, scoreAverage, scoreSTD]
alg = tree.DecisionTreeClassifier(min_samples_leaf = optimalTreeSamples, max_depth= optimalTreeDepth, random_state=0)
scores = cross_val_score(alg, X, y, cv = 5)
scoreAverage = scores.mean()
print('Final DecisionTree Score: {:01.5f}'.format(scoreAverage))
scoreSTD = scores.std() * 2
finalScoreFrame.loc[len(finalScoreFrame)] = [alg.__class__.__name__, scoreAverage, scoreSTD]
alg = neighbors.KNeighborsClassifier(n_neighbors = optimalKNNNeighbors, weights = optimalKNNWeight, algorithm = optimalKNNAlg)
scores = cross_val_score(alg, X, y, cv = 5)
scoreAverage = scores.mean()
print('Final KNC Score: {:01.5f}'.format(scoreAverage))
scoreSTD = scores.std() * 2
finalScoreFrame.loc[len(finalScoreFrame)] = [alg.__class__.__name__, scoreAverage, scoreSTD]
alg = ensemble.ExtraTreesClassifier(n_estimators=optimalExtraTreeN, max_depth=optimalExtraTreeDepth, random_state=0)
scores = cross_val_score(alg, X, y, cv = 5)
scoreAverage = scores.mean()
print('Final Extra Trees Score: {:01.5f}'.format(scoreAverage))
scoreSTD = scores.std() * 2
finalScoreFrame.loc[len(finalScoreFrame)] = [alg.__class__.__name__, scoreAverage, scoreSTD]
alg = ensemble.RandomForestClassifier(n_estimators=optimalRandomForestN, max_depth=optimalRandomForestDepth, min_samples_split=optimalRandomForestSamples, random_state=0)
scores = cross_val_score(alg, X, y, cv = 5)
scoreAverage = scores.mean()
print('Final Random Forest Score: {:01.5f}'.format(scoreAverage))
scoreSTD = scores.std() * 2
finalScoreFrame.loc[len(finalScoreFrame)] = [alg.__class__.__name__, scoreAverage, scoreSTD]
alg = ensemble.GradientBoostingClassifier(n_estimators = optimalGBCN, learning_rate = optimalGBCLearningRate, max_depth = optimalGBCDepth, random_state = 0)
scores = cross_val_score(alg, X, y, cv = 5)
scoreAverage = scores.mean()
print('Final Gradient Boosting Score: {:01.5f}'.format(scoreAverage))
scoreSTD = scores.std() * 2
finalScoreFrame.loc[len(finalScoreFrame)] = [alg.__class__.__name__, scoreAverage, scoreSTD]
alg = ensemble.AdaBoostClassifier(n_estimators = optimalAdaBoostN, base_estimator=optimalAdaBoostBase, learning_rate=optimalAdaBoostLearningRate, random_state = 0)
scores = cross_val_score(alg, X, y, cv = 5)
scoreAverage = scores.mean()
print('Final AdaBoost Score: {:01.5f}'.format(scoreAverage))
scoreSTD = scores.std() * 2
finalScoreFrame.loc[len(finalScoreFrame)] = [alg.__class__.__name__, scoreAverage, scoreSTD]
compareScoreFrame = pd.DataFrame(columns = ['Algorithm Name', 'Average', 'Standard Deviation', 'Before/After'])

for i in range(len(scoreFrame)):
    row = scoreFrame.loc[i]
    compareScoreFrame.loc[len(compareScoreFrame)] = [row['Algorithm Name'], row['Average'], row['Standard Deviation'], 'Before']
    
for i in range(len(finalScoreFrame)):
    row = finalScoreFrame.loc[i]
    compareScoreFrame.loc[len(compareScoreFrame)] = [row['Algorithm Name'], row['Average'], row['Standard Deviation'], 'After']


compareScoreFrame.sort_values('Average', ascending=False).head(3)
compareScoreFrame.sort_values('Average', ascending=False)
g = sns.relplot(x = "Algorithm Name", y = "Average", hue="Before/After", size = 'Standard Deviation', data = compareScoreFrame, sizes = (100,500), height=7) 
g.fig.autofmt_xdate()