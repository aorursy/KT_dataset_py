import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
eps = np.finfo(float).eps
from numpy import log2 as log

plt.style.use("fivethirtyeight")
sns.set_style({'font.sans-serif':['simhei','Arial']})
%matplotlib inline

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sys import version_info
if version_info.major != 3:
    raise Exception
bus_stops = pd.read_csv('/kaggle/input/barcelona-data-sets/bus_stops.csv')

display(bus_stops.head(n=10))
bus_stops.info()
bus_stops.describe()
df = bus_stops.copy()


columns = ['Code','Transport','Longitude','Latitude','Bus.Stop','District.Name','Neighborhood.Name']
df = pd.DataFrame(df)


display(df.head(n=5))
print(f"Totally, there are {len(bus_stops)} records")

bus_stops.head()
for k in bus_stops.keys():
    print(k)
# State
# County
df.describe()
Transport = 'Day,Day,Night,Day,Night,Night,Day,Day,Day,Day,Night,Day,Day,Day,Night,Day,Day,Day,Day,Night,Day,Day,Day,Night,Night,Day,Day,Day,Night,Day,Day,Day,Day,Day,Night,Night,Day,Day,Day,Day,Night,Day,Day,Day,Day,Night,Day,Day,Day,Day'.split(',')
longitude = '214,214,216,214,214,222,216,214,222,214,222,214,222,216,216,214,216,222,214,222,222,214,214,214,222,216,222,216,222,216,216,222,214,216,222,216,222,214,216,216,216,222,216,214,222,216,216,216,216,216'.split(',')
latitude = '414,413,415,413,415,413,413,413,413,414,415,413,413,414,415,414,413,415,413,415,414,413,413,413,415,415,413,415,413,413,413,413,414,415,413,415,413,414,415,414,414,415,414,415,415,415,415,415,415,415'.split(',')
Code = 'K0141,K015,K0141,K0141,K0141,K015,K0141K015,K015,K0141,K0141,K0141,K015,K0141,K0141,K0141,K0141,K0141,K015,K015,K0141,K0141,K0141,K015,K0141,K0141,K0141,K0141,K0141,K015,K015,K0141,K0141,K0141,K0141,K015,K0141,K015,K0141,K0141,K0141,K015,K0141,K0141,K0141,K0141,K0141,K0141,K0141,K0141,'.split(',')
GoodJourney = 'yes,yes,yes,yes,no,yes,yes,yes,yes,yes,no,no,yes,yes,yes,yes,yes,yes,yes,yes,yes,yes,yes,yes,yes,yes,no,yes,no,yes,yes,yes,no,yes,yes,yes,yes,no,yes,yes,yes,no,yes,no,no,yes,yes,yes,yes,yes'.split(',')
dataset ={'Transport':Transport,'longitude':longitude,'latitude':latitude,'Code':Code,'GoodJourney':GoodJourney}
df = pd.DataFrame(dataset,columns=['Transport','longitude','latitude','Code','GoodJourney'])
##1. claculate entropy o the whole dataset

entropy_node = 0  #Initialize Entropy
values = df.GoodJourney.unique()  #Unique objects - 'Yes', 'No'
for value in values:
    fraction = df.GoodJourney.value_counts()[value]/len(df.GoodJourney)  
    entropy_node += -fraction*np.log2(fraction)
print(entropy_node) #Still get the same value as the above

def ent(df,attribute):
    target_variables = df.GoodJourney.unique()  #This gives all 'Yes' and 'No'
    variables = df[attribute].unique()    #This gives different features in that attribute 


    entropy_attribute = 0
    for variable in variables:
        entropy_each_feature = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute]==variable][df.GoodJourney ==target_variable]) #numerator
            den = len(df[attribute][df[attribute]==variable])  #denominator
            fraction = num/(den+eps)  #pi
            entropy_each_feature += -fraction*log(fraction+eps) #This 
        fraction2 = den/len(df)
        entropy_attribute += -fraction2*entropy_each_feature   #Sums up all the entropy 

    return(abs(entropy_attribute))
a_entropy = {k:ent(df,k) for k in df.keys()[:-1]} #store entropy of each attribute with its name
a_entropy
def ig(e_dataset,e_attr):
    return(e_dataset-e_attr)
#entropy_node = entropy of dataset
#a_entropy[k] = entropy of k(th) attr
IG = {k:ig(entropy_node,a_entropy[k]) for k in a_entropy}
print(IG)
def cal_entropy(y):
    """Information entropy calculation
    
     parameter 
     y: category number type: narray, shape: {n_samples}
    
     return
     e: Information entropy type: float
    """
    count = np.array(pd.Series(y).value_counts())
    p = count/count.sum()
    return -np.sum(np.log2(p)*p)
def choose_features_ID3(X, y):
    """Select feature (single)
    parameter
    X: features, type: ndarray, shape: {n_samples, n_features}
    y: category number type: narray, shape: {n_samples}
    return
    min_fea_index: selected feature, type: integer
    entropy: information gain, type: float
    """
    n_samples, n_features = X.shape
    
    fea_index = 0
    max_entropy = 0
    pre_y_entropy = cal_entropy(y)
    for i in range(n_features):
        entropy_sum = 0
        row_value = X[:,i]
        for value in set(row_value):
            bools = row_value==value
            entropy_sum += np.sum(bools)/n_samples * cal_entropy(y[bools])
        entropy = pre_y_entropy-entropy_sum
        if entropy>max_entropy:
            max_entropy = entropy
            fea_index = i
    return fea_index,entropy
def tree_ID3(X, y, X_name):
    """Establish decision tree, adopt ID3, no pruning operation
    parameter
    X: features, type: ndarray, shape: {n_samples, n_features}
    y: category number, type: ndarray, shape: {n_samples} 
    X_name: feature name, type: ndarray, shape: {n_samples}
    """
    if not len(X):return 
    if cal_entropy(y)==0:return y[0]
    
    n_samples, n_features = X.shape
    index = choose_features_ID3(X, y)[0]
    dic = {X_name[index]:{}}
    remove_fea = X[:, index]
    for fea in set(remove_fea):
        row_bool = remove_fea==fea  # Row index
        col_bool = np.arange(n_features)!=index   # Row index
        dic[X_name[index]][fea] = tree_ID3(X[row_bool][:,col_bool], y[row_bool], X_name[col_bool])
    return dic
dataSet = np.array([
        
        ['Green','Curled up','Turbid sound','Clear','Depressed','Hard and slippery','Good melon'],
        ['Black','Curled up','Dull','Clear','Depressed','Hard and slippery','Good melon'],
        ['Black','Curled up','Turbid sound','Clear','Depressed','Hard and slippery','Good melon'],
        ['Green','Curled up','Dull','Clear','Depressed','Hard and slippery','Good melon'],
        ['Light white','Curled up','Turbid sound','Clear','Depressed','Hard and slippery','Good melon'],
        ['Green','Slightly curled up','Turbid sound','Clear','Slightly concave','Soft sticky','Good melon'],
        ['Black','Slightly curled up','Turbid sound','Slightly muddy','Slightly concave','Soft sticky','Good melon'],
        ['Black','Slightly curled up','Turbid sound','Clear','Slightly concave','Hard and slippery','Good melon'],
        ['Black','Slightly curled up','Dull','Slightly muddy','Slightly concave','Hard and slippery','Bad melon'],
        ['Green','Stiff','Crispy','Clear','Flat','Soft sticky','Bad melon'],
        ['Light white','Stiff','Crisp','Fuzzy','Flat','Hard and slippery','Bad melon'],
        ['Light white','Curled up','Turbid sound','Fuzzy','Flat','Soft sticky','Bad melon'],
        ['Green','Slightly curled up','Turbid sound','Slightly muddy','Depressed','Hard and slippery','Bad melon'], 
        ['Light white','Slightly curled up','Dull','Slightly muddy','Depressed','Hard and slippery','Bad melon'],
        ['Black','Slightly curled up','Turbid sound','Clear','Slightly concave','Soft sticky','Bad melon'],
        ['Light white','Curled up','Turbid sound','Fuzzy','Flat','Hard and slippery','Bad melon'],
        ['Green','Curled up','Dull','Slightly muddy','Slightly concave','Hard and slippery','Bad melon']
    ])
X = dataSet[:,:-1]
y = dataSet[:,-1]
X_name = np.array(['color','root','knock','texture','umbilical','touch'])
tree_ID3(X,y,X_name)

class Tree_ID3:
    def __init__(self):
        pass
        
    def cal_entropy(self, y):
        count = np.array(pd.Series(y).value_counts())
        # Probability of each category
        p = count/count.sum()
        # Information Entropy
        return -np.sum(np.log2(p)*p)

    def choose_features_ID3(self, X, y):
        n_samples, n_features = X.shape

        # Optimal feature index
        fea_index = 0
        # Maximum information gain
        max_entropy = 0
        # Information entropy of label y before classification
        pre_y_entropy = self.cal_entropy(y)
        
        for i in range(n_features):
            # Initialize the weighted information entropy after classification
            entropy_sum = 0
            row_value = X[:,i]
            for value in set(row_value):
                # Selected sample index
                bools = row_value==value
                entropy_sum += np.sum(bools)/n_samples * self.cal_entropy(y[bools])
            # Current information gain
            entropy = pre_y_entropy-entropy_sum
            if entropy>max_entropy:
                max_entropy = entropy
                fea_index = i
        return fea_index,entropy

    def tree_ID3(self, X, y, X_name):
        if not len(X):return
        # Only one category left, return
        if self.cal_entropy(y)==0:return y[0]

        n_samples, n_features = X.shape
        index = self.choose_features_ID3(X, y)[0]
        # Decision tree construction
        dic = {X_name[index]:{}}
        remove_fea = X[:, index]
        for fea in set(remove_fea):
            # Remaining row index
            row_bool = remove_fea==fea
            # Remaining column index
            col_bool = np.arange(n_features)!=index
            # Recursion
            dic[X_name[index]][fea] = self.tree_ID3(X[row_bool][:,col_bool], y[row_bool], X_name[col_bool])
        return dic
    
    def check(self, tree, X, X_name):
        """prediction
        """
        if not len(tree) or not len(X):return
        cur_fea_name = list(tree.keys())[0]
        cur_fea_index = np.where(X_name==cur_fea_name)[0][0]
        if X[cur_fea_index] not in tree[cur_fea_name].keys():return
        if tree[cur_fea_name][X[cur_fea_index]] in self.y_name:
            return tree[cur_fea_name][X[cur_fea_index]]
        else:
            bools = np.arange(len(X))!=cur_fea_index
            return self.check(tree[cur_fea_name][X[cur_fea_index]], X[bools], X_name[bools])
    
    def fit(self, X, y, X_name):
        self.X_name = X_name
        self.y_name = list(set(y))
        self.tree = self.tree_ID3(X, y, X_name)
        
    def predict(self, X):
        res = []
        for i in range(len(X)):
            res.append(self.check(self.tree, X[i], self.X_name))
        return np.array(res)
clf = Tree_ID3()
clf.fit(X, y, X_name)
predict_y = clf.predict(X)
sum(predict_y==y)==len(y)

dataSet = np.array([
['K0141','Day','214','414','yes'],                       
['K015','Day','214','413','yes'],
['K0141','Night','216','415','yes'],
['K0141','Day','214','413','yes'],
['K0141','Night','214','415','no'],
['K015','Night','222','413','yes'],
['K0141','Day','216','413','yes'],
['K015','Day','214','413','yes'],
['K015','Day','222','413','yes'],
['K0141','Day','214','414','yes'],    
['K0141','Night','222','415','no'],
['K0141','Day','214','413','no'],
['K015','Day','222','413','yes'],
['K0141','Day','216','414','yes'],
['K0141','Night','216','415','yes'],
['K0141','Day','214','414','yes'],
['K0141','Day','216','413','yes'],
['K0141','Day','222','415','yes'],
['K015','Day','214','413','yes'],
['K015','Night','222','415','yes'],    
['K0141','Day','222','414','yes'],
['K0141','Day','214','413','yes'],
['K0141','Day','214','413','yes'],
['K015','Night','214','413','yes'],
['K0141','Night','222','414','yes'],
['K0141','Day','216','415','yes'],
['K0141','Day','222','413','no'],
['K0141','Day','216','415','yes'],
['K0141','Night','222','413','no'],
['K015','Day','216','413','yes'],   
['K015','Day','216','413','yes'],
['K0141','Day','222','413','yes'],
['K0141','Day','214','414','no'],
['K0141','Day','216','415','yes'],
['K0141','Night','222','413','yes'],
['K015','Night','216','415','yes'],
['K0141','Day','222','413','yes'],
['K015','Day','214','414','no'],
['K0141','Day','216','415','yes'],
['K0141','Day','216','414','yes'],
['K0141','Night','216','414','yes'],
['K015','Day','222','415','no'],
['K0141','Day','216','414','yes'],
['K0141','Day','214','415','no'],
['K0141','Day','222','415','no'],
['K0141','Night','216','415','yes'],
['K0141','Day','216','415','yes'],
['K0141','Day','216','415','yes'],
['K0141','Day','216','415','yes'],
['K0141','Day','216','415','yes'],
     ])
X = dataSet[:,::-1]
y = dataSet[:,-1]
X_name = np.array(['Code','Transport','longitude','latitude','GoodJourney'])
tree_ID3(X,y,X_name)
clf = Tree_ID3()
clf.fit(X, y, X_name)
predict_y = clf.predict(X)
sum(predict_y==y)==len(y)