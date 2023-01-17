import numpy as np # linear algebra
import pandas as pd
import keras
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.losses import *
from keras.losses import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head()
data['sentence'] = data.apply(lambda row: 'gender:'+str(row['gender'])+'|'
                              + 'SeniorCitizen:'+str(row['SeniorCitizen'])+'|'
                              + 'Partner:'+str(row['Partner'])+'|'
                              + 'Dependents:'+str(row['Dependents'])+'|'
                              + 'PhoneService:'+str(row['PhoneService'])+'|'
                              + 'MultipleLines:'+str(row['MultipleLines'])+'|'
                              + 'InternetService:'+str(row['InternetService'])+'|'
                              + 'OnlineSecurity:'+str(row['OnlineSecurity'])+'|'
                              + 'OnlineBackup:'+str(row['OnlineBackup'])+'|'
                              + 'DeviceProtection:'+str(row['DeviceProtection'])+'|'
                              + 'TechSupport:'+str(row['TechSupport'])+'|'
                              + 'StreamingTV:'+str(row['StreamingTV'])+'|'
                              + 'StreamingMovies:'+str(row['StreamingMovies'])+'|'
                              + 'Contract:'+str(row['Contract'])+'|'
                              + 'PaperlessBilling:'+str(row['PaperlessBilling'])+'|'
                              + 'PaymentMethod:'+str(row['PaymentMethod'])
                              ,axis=1)
data_proc = data[['customerID','sentence','tenure','MonthlyCharges','TotalCharges','Churn']]
data_proc['TotalCharges'] = data_proc['TotalCharges'].apply(lambda x: float(x) if x!=' ' else 0.0)
data_proc.head()
sns.countplot(x="Churn", data=data);
X_cat = data_proc['sentence'].values
X_num = StandardScaler().fit_transform(data_proc[['tenure','MonthlyCharges','TotalCharges']].astype(np.float).values)
y = data_proc['Churn'].apply(lambda x: 1 if x=='Yes' else 0).values

tokenizer = Tokenizer(filters='',lower=False,split='|')
tokenizer.fit_on_texts(X_cat)
X_cat_seq = pad_sequences(tokenizer.texts_to_sequences(X_cat))
SEQ_LEN = len(X_cat_seq[0])
MAX_ID = np.max(X_cat_seq)
def make_model():
    cat_in = Input((SEQ_LEN,))
    num_in = Input((3,))
    embedding = Embedding(input_dim=MAX_ID+1, output_dim=20)(cat_in)
    x = SpatialDropout1D(0.5)(embedding)
    x = GlobalAveragePooling1D()(x)
    x = concatenate([x, num_in])
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(cat_in, embedding), Model([cat_in, num_in], x)
embedding, model = make_model()
model.summary()

X_cat_seq_train,X_cat_seq_test,X_num_train,X_num_test,y_train,y_test = train_test_split(X_cat_seq, X_num, y, test_size=0.1)
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])
history = model.fit([X_cat_seq_train, X_num_train], y_train,
                    epochs=100,
                    validation_data=([X_cat_seq_test, X_num_test], y_test),
                    callbacks=[EarlyStopping(monitor='val_loss',patience=1,verbose=2)],
                    verbose=2)
    
y_pred = np.round(model.predict([X_cat_seq_test, X_num_test]))
    
#plt.figure()
#plt.plot(history.history['loss'], 'r')
#plt.plot(history.history['val_loss'], 'b')
    
#plt.figure()
#plt.plot(history.history['acc'], 'r')
#plt.plot(history.history['val_acc'], 'b')
    
plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
    
print(classification_report(y_test, y_pred))
keys = list(tokenizer.word_index.keys())
key_seq = tokenizer.texts_to_sequences(keys)
vecs = embedding.predict(pad_sequences(key_seq,SEQ_LEN,padding='post'))

plt.figure(figsize=(20,20))
x = []
y = []
for v in vecs:
    x.append(v[0][0])
    y.append(v[0][1])
plt.scatter(x,y)
for i,k in enumerate(keys):
    plt.text(x[i],y[i],k)