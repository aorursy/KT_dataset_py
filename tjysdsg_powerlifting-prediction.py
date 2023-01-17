import pandas as pd
import numpy as np
data = pd.read_csv("../input/openpowerlifting.csv")
data.drop(["Squat4Kg","Bench4Kg","Deadlift4Kg"],axis=1,inplace=True)
data.drop_duplicates(inplace=True)
display(data.head())
display(data.describe())
data_cleaned = data[(data['BestSquatKg'] > 0)&(data['BestBenchKg']>0)&(data['BestDeadliftKg']> 0)]
# encode
data_cleaned=data_cleaned.dropna()
data_cleaned['Sex'] = data_cleaned['Sex'].map( {'M':1, 'F':0} )
# encode
from sklearn.preprocessing import LabelEncoder
data_cleaned['Equipment'] = LabelEncoder().fit_transform(data_cleaned['Equipment'])
#
data_cleaned['WeightClassKg']=data_cleaned['WeightClassKg'].str.extract('(^\d*)') 
data_cleaned.describe()
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
fig,ax = plt.subplots(3,3)
fig.set_size_inches(20, 20)
feature_names = ['Age','BodyweightKg','Wilks']
outcome_names = ['BestSquatKg','BestBenchKg','BestDeadliftKg']
for i in range(len(feature_names)):
    for j in range(len(outcome_names)):
        ax[i,j].plot(data_cleaned[feature_names[i]],data_cleaned[outcome_names[j]],'p')
        ax[i,j].set_title(outcome_names[j]+"~"+feature_names[i])
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from keras.activations import *
from keras.optimizers import *
from keras.models import *
from keras.layers import *
#feature_names = ['Age','BodyweightKg', 'Equipment','Wilks','WeightClassKg','Sex']
#outcome_names = ['BestSquatKg','BestBenchKg','BestDeadliftKg']

def fit_with_features(features,outcomes):
    # features
    X = data_cleaned[features]
    # rescale
    scaler_X = MinMaxScaler(feature_range=(0, 1)) 
    X = scaler_X.fit_transform(X)
    # outcomes
    y = data_cleaned[outcomes]
    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
    """
    best run suing hyperas:
     {'Dense': 0, 'Dropout': 0.5604510956047549, 'activation': 1, 'activation_1': 1, 'batch_size': 1, 'loss': 0, 'optimizer': 1}
    
    The choices are:
    
    Neuron count of the first layer: np.power(2, 5), np.power(2, 6), np.power(2, 7)
    Activation funciton of the first layer: 'sigmoid','relu'
    Dropout Layer: uniform(0.5,1)
    Activation function of the second layer: 'sigmoid','relu'
    optimizer: 'rmsprop', 'adam', 'sgd'
    loss: "mean_squared_error", "binary_crossentropy"
    batch_size: 64, 128]
    """
    model = Sequential()
    model.add(Dense(np.power(2, 5), input_dim=X_train.shape[1], activation='sigmoid'))
    model.add(Dropout(0.5604510956047549))
    model.add(Dense(y_train.shape[1], activation='relu'))
    model.compile(optimizer= 'adam',
                  loss="mean_squared_error",
                  metrics=['acc'])
    
    result = model.fit(X_train, y_train,batch_size=64,epochs=2,verbose=0,validation_split=0.1)
    evaluation = model.evaluate(X_test,y_test)
    print("Accuracy:",evaluation[1])
fit_with_features(['Age','BodyweightKg', 'Equipment','Wilks','WeightClassKg','Sex'],['BestSquatKg','BestBenchKg','BestDeadliftKg'])
fit_with_features(['Age','BodyweightKg', 'Equipment','WeightClassKg','Sex'],['BestSquatKg','BestBenchKg','BestDeadliftKg'])
fit_with_features(['Age','BodyweightKg', 'Equipment','Wilks','Sex'],['BestSquatKg','BestBenchKg','BestDeadliftKg'])
fit_with_features(['Age','BodyweightKg','Wilks','Sex'],['BestSquatKg','BestBenchKg','BestDeadliftKg'])
#bar mean
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
data_tract=data_cleaned['BestSquatKg','BestBenchKg','BestDeadliftKg']
listnames=list['BestSquatKg','BestBenchKg','BestDeadliftKg']
data_mean=np.mean(data_tract)
plt.bar(listnames,data_mean)
plt.title("Weightlift mean")
plt.show()
#bar mean separately
data_merge1=list[data.cleaned['Division']]
data_merge2=[]
for data in data_merge1:
    if data not in data_merge1:
        data_merge2=data_merge2.append(data)
print (len(data_merge2))
#distribution
sns.violinplot(df['Age'], df['WeightClass']) #Variable Plot
sns.despine()
#pie
var=df.groupby(['Division']).sum().stack()
temp=var.unstack()
type(temp)
x_list = temp['BodyWeight']
label_list = temp.index
pyplot.axis("equal") 
plt.pie(x_list,labels=label_list,autopct="%1.1f%%") 
plt.title("Division distribution")
plt.show()
#Pie2
var=df.groupby(['Division']).sum().stack()
temp=var.unstack()
type(temp)
x_list = temp['WeightClass']
label_list = temp.index
pyplot.axis("equal") 
plt.pie(x_list,labels=label_list,autopct="%1.1f%%") 
plt.title("Division distribution")
plt.show()




