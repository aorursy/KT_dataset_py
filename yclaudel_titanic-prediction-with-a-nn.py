import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import time

import re



from pylab import rcParams

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score, recall_score



sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 10, 6

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)



def print_metrics(y_true,y_pred):

    conf_mx = confusion_matrix(y_true,y_pred)

    print ("------------------------------------------")

    print (" Accuracy    : ", accuracy_score(y_true,y_pred))

    print (" Precision   : ", precision_score(y_true,y_pred))

    print (" Sensitivity : ", recall_score(y_true,y_pred))

    print ("------------------------------------------")

    print(classification_report(y_true, y_pred))

    print ("------------------------------------------")

    class_names = [0,1]

    fig,ax = plt.subplots()

    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks,class_names)

    plt.yticks(tick_marks,class_names)

    

    sns.heatmap(pd.DataFrame(conf_mx),annot=True,cmap="Blues",fmt="d",cbar=False)

    ax.xaxis.set_label_position('top')

    plt.tight_layout()

    plt.ylabel('Actual label')

    plt.xlabel('Predicted label')

    plt.show()
######################################################

# load data

######################################################

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

submit_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')



train_df.head(10)
######################################################

# features

######################################################

def feature_eng(df,columnshoice):

    df['TicketLetter'] = df['Ticket'].apply(lambda x : str(x)[0]) 

    df['TicketLetter'] = df['TicketLetter'].apply(lambda x : re.sub('[0-9]','N',x))

    df['Title'] = df.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

    normalized_titles = {

        "Capt":       "Officer",

        "Col":        "Officer",

        "Major":      "Officer",

        "Jonkheer":   "Royalty",

        "Don":        "Royalty",

        "Sir" :       "Royalty",

        "Dr":         "Officer",

        "Rev":        "Officer",

        "the Countess":"Royalty",

        "Dona":       "Royalty",

        "Mme":        "Mrs",

        "Mlle":       "Miss",

        "Ms":         "Mrs",

        "Mr" :        "Mr",

        "Mrs" :       "Mrs",

        "Miss" :      "Miss",

        "Master" :    "Master",

        "Lady" :      "Royalty"

    }



    df['Title'] = df['Title'].map(normalized_titles)

    df['NameLength'] = df['Name'].apply(lambda x : len(x))

    df['NameLength'] = ((df.NameLength)/15).astype(np.int64)+1



    df["FamilySize"] = df['SibSp'] + df['Parch'] + 1

    # bins = [-1,1, 2, 3,4, np.inf]

    # labels = ['ONE','TWO', 'THREE', 'FOUR','BIG']

    bins = [-1,1,4, np.inf]

    labels = ['ONE','SMALL','BIG']

    df['FamilyGroup'] = pd.cut(df["FamilySize"], bins, labels = labels)

    df['IsAlone'] = 'Y'

    df.loc[df['FamilySize'] > 1,'IsAlone'] = 'N'

    df["Embarked"] = df["Embarked"].fillna("S")

    df["Age"] = df.groupby(['Sex','Title'])["Age"].transform(lambda x: x.fillna(x.median()))

    df["Cabin"] = df["Cabin"].str[0:1]

    df["Cabin"] = df["Cabin"].fillna('T')

    df['Fare'] = df['Fare'].fillna(-1)





    return df[columnshoice]



columnshoice = ['Pclass', 'Sex', 'Age', 'SibSp',

           'Parch', 'Fare', 'Embarked', 'TicketLetter', 'Title',

           'FamilySize', 'IsAlone','NameLength']





y = train_df["Survived"]

X = feature_eng(train_df,columnshoice)

test = feature_eng(test_df,columnshoice)

print(X.columns)



# pd.get_dummies(df[columnshoice])





print(X.shape)



from sklearn import preprocessing

numerical_features = list(X.select_dtypes(include=['int64', 'float64', 'int32']).columns)

scaler = preprocessing.StandardScaler()

X[numerical_features] = scaler.fit_transform(X[numerical_features])

test[numerical_features] = scaler.transform(test[numerical_features])



X = pd.get_dummies(X)

test = pd.get_dummies(test)
#########################################

# Build model

#########################################

model = tf.keras.models.Sequential([

  tf.keras.layers.Dense(units=64, activation='relu',input_dim=X.shape[1]),

  # tf.keras.layers.Dropout(rate=0.2),

  tf.keras.layers.Dense(units=32, activation='relu'),

  tf.keras.layers.Dense(units=32, activation='relu'),

  tf.keras.layers.Dense(units=1, activation='sigmoid')

])

#Visualizing the model

#model.summary()



#  Stop training early

early_stop = keras.callbacks.EarlyStopping(

  monitor='val_loss',

  patience=5

)



model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])





# Fit the model

history = model.fit(

  x=X,

  y=y,

  shuffle=True,

  #batch_size = 30,

  epochs=100,

  validation_split=0.1,

  verbose=0,

  callbacks=[early_stop]

)
#########################################

# Evaluate model

#########################################



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.ylim((0, 1))

plt.legend(['train', 'test'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
######################################################

# predictions

######################################################

predictions = model.predict(X)

predictions = tf.round(predictions).numpy().flatten().astype(int)
# Precision : among positive test, % of real value

# Sensitivity : the probability that the test is positive given that the subject is really positive

# Specificity (recall of 0) : the probability that the test is negative given that the subject is really negative



print_metrics(y,predictions)

######################################################

# submit

######################################################

y_sub= model.predict(test)

y_sub = tf.round(y_sub).numpy().flatten().astype(int)

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": y_sub

    })

submission.to_csv('titanic.csv', index=False)