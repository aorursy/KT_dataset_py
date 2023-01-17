import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns

dataset = pd.read_csv('../input/titanic/train.csv')

X = dataset.iloc[:, 2:]
y = dataset.iloc[:, 1:2]
X.info()
X = X.drop(['Name'], axis = 1)
X = X.drop(['Ticket'], axis = 1)
#X = X.drop(['Cabin'], axis = 1)

X['Age'] = X['Age'].fillna(X['Age'].mean())
X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])
import math

decks = []


for i in range(len(X)):
    value = X['Cabin'][i]
    if not pd.isnull(value) and value != 'T':
        # this is to deal with some strange data e.g. "F G63"
        if len(value) > 2 and value[1] == ' ':
            deck = value[2]
            decks.append(deck)
        else:
            # this is to handle the normal case
            deck = value[0]
            decks.append(deck)
    else:
        decks.append("")
        
decks = pd.DataFrame(decks, columns = ['Deck'])

X = pd.concat([X,decks],axis=1)

X = X.drop(['Cabin'], axis = 1)




X.head(25)
def oneHotEncoder(dataframe):
    
    categ = []

    for i in dataframe.columns:

      if dataframe[i].dtype == np.object:

        categ.append(i)

    df_final = dataframe
    
    i=0
    for field in categ:
        
        df1=pd.get_dummies(dataframe[field], drop_first = True)
        
        dataframe.drop([field],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final, df1],axis=1)
        i += 1
       
        
    df_final=pd.concat([dataframe,df_final],axis=1)
        
    return df_final
X2 = oneHotEncoder(X)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.2, random_state = 1)



## Feature Scaling

columns = X_train.columns

#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = pd.DataFrame(data=X_train, columns = columns)
X_test = pd.DataFrame(data=X_test, columns = columns)


"""
# input column numbers here, either normally distributed or not
normalDistCols = []
uniformDistCols = ['Pclass','Age', 'SibSp', 'Parch', 'Fare']


for i in normalDistCols:
    
    sc = StandardScaler()
    X_train[:, i:i+1] = sc.fit_transform(X_train[:, i:i+1])
    X_test[:, i:i+1] = sc.transform(X_test[:, i:i+1])
    
    
    
for j in uniformDistCols:
    
    
    sc = MinMaxScaler(copy = False)
    X_train[[j]] = sc.fit_transform(X_train[[j]])
    X_test[[j]] = sc.transform(X_test[[j]])
    
"""
    
    
X_train.head(15)
import tensorflow as tf

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units = 25, activation = 'sigmoid'))
ann.add(tf.keras.layers.Dense(units = 25, activation = 'sigmoid'))
ann.add(tf.keras.layers.Dense(units = 25, activation = 'sigmoid'))





ann.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy')

epochs = 20


history = ann.fit(X_train, y_train, batch_size = 32, epochs = epochs, validation_data=(X_test, y_test))

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochRange = range(1, epochs+1)
plt.plot(epochRange, loss_train, 'g', label='Training loss')
plt.plot(epochRange, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

from keras import backend as K
K.set_value(ann.optimizer.learning_rate, 0.001)

epochs = 4


history = ann.fit(X_train, y_train, batch_size = 32, epochs = epochs, validation_data=(X_test, y_test))

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochRange = range(1, epochs+1)
plt.plot(epochRange, loss_train, 'g', label='Training loss')
plt.plot(epochRange, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
import xgboost
xgbclassifier=xgboost.XGBClassifier()


#base_score=[0.25,0.5,0.75,1]

n_estimators = [200,250,300, 350]
max_depth = [2,3,4,5]
booster=['gbtree']
learning_rate=[0.01, 0.02, 0.03, 0.05]
min_child_weight=[2, 3, 4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster
    #'base_score':base_score
    }

from sklearn.model_selection import GridSearchCV

grid_cv = GridSearchCV(xgbclassifier, 
                       hyperparameter_grid,
                       n_jobs = -1,
                       cv=5, 
                       scoring='accuracy', 
                       verbose = 5,
                       return_train_score=True)

grid_cv.fit(X_train, y_train)

print(grid_cv.best_estimator_)
print(grid_cv.best_params_)
print(grid_cv.best_score_)


test_dataset = pd.read_csv('../input/titanic/test.csv')

X = test_dataset.iloc[:,1:]

X = X.drop(['Name'], axis = 1)
X = X.drop(['Ticket'], axis = 1)
#X = X.drop(['Cabin'], axis = 1)

X['Age'] = X['Age'].fillna(X['Age'].mean())
X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])


decks = []

for i in range(len(X)):
    value = X['Cabin'][i]
    if not pd.isnull(value) and value != 'T':
        # this is to deal with some strange data e.g. "F G63"
        if len(value) > 2 and value[1] == ' ':
            deck = value[2]
            decks.append(deck)
        else:
            # this is to handle the normal case
            deck = value[0]
            decks.append(deck)
    else:
        decks.append("")
        
decks = pd.DataFrame(decks, columns = ['Deck'])

X = pd.concat([X,decks],axis=1)

X = X.drop(['Cabin'], axis = 1)


X2 = oneHotEncoder(X)

X2 = sc.transform(X2)
X2 = pd.DataFrame(data=X2, columns = columns)


    
X2.head(15)
y_pred = grid_cv.predict(X2)

# probabilities (if needed)
#y_pred_prob = ann.predict(X2)

y_pred
pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('../input/titanic/gender_submission.csv')
datasets=pd.concat([sub_df['PassengerId'],pred],axis=1)
datasets.columns=['PassengerId','Survived']
datasets.to_csv('submission.csv',index=False)