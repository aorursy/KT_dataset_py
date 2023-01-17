#First we need import pandas to create dataframes, Matplotlib to visualize our data and numpy to create and reshape arrays.
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
%matplotlib inline  
#A class to analyze punctuations from a text
class PunctuationAnalyzer():
    
    #function that counts punctuation
    def count_punctuation(self,text):

        punctuation_list={"Ellipsis Points":0,"Dot":0,"Comma":0,"Colon":0,"Dash":0,"Exclamation Mark":0,"Question Mark":0,"Semi-colon":0,"Parantheses":0,"Apostrophe":0,"Quotation Marks":0}
        char=0
        for i in text:
            elnot=True
            if(char+2<len(text)):
                if((i+text[char+1]+text[char+2])=="..." or i=="…" or i+text[char+1]==".."):
                    
                    punctuation_list["Ellipsis Points"]+=1
                    elnot=False
                    
            if(i=="." and elnot):
                punctuation_list["Dot"]+=1
            elif(i==","):
                punctuation_list["Comma"]+=1
            elif(i==":"):
                punctuation_list["Colon"]+=1
            elif(i=="-"):
                punctuation_list["Dash"]+=1
            elif(i=="!"):
                punctuation_list["Exclamation Mark"]+=1
            elif(i=="?"):
                punctuation_list["Question Mark"]+=1
            elif(i==";"):
                punctuation_list["Semi-colon"]+=1
            elif(i=="("):
                punctuation_list["Parantheses"]+=1
            elif(i=="'"):
                punctuation_list["Apostrophe"]+=1
            elif(i=='"'):
                 punctuation_list["Quotation Marks"]+=1     
            char+=1
        return punctuation_list
    
    #function that returns data and labels to create data frames
    def punct_forFrame(self,punct_dict):
        data=[]
        labels=list()
        for k,v in punct_dict.items():
            l=[v]
            data.append(l)
            labels.append(k)
        return data,labels
    
    def to_Frame(self,işaretler):
        data,labels=pa.punct_forFrame(işaretler)
        data=np.array(data)
        data=data.reshape(1,11)
        
        return pd.DataFrame(data,columns=labels)

#a function that read a file
def read_file(path):
    with open(path,"r") as file:

        return file.read()
#Reading the files
text1=read_file("../input/Montaigne-JoA.txt")
text2=read_file("../input/Montaigne-OfIdleness.txt")
text3=read_file("../input/Montaigne-OfLiars.txt")
text4=read_file("../input/Montaigne-OfSorrow.txt")
text5=read_file("../input/Montaigne-PD.txt")

#Creating the PunctuationAnalyzer object
pa=PunctuationAnalyzer()
#We analyze the punctuations here
Montaigne1=pa.count_punctuation(text1)
Montaigne2=pa.count_punctuation(text2)
Montaigne3=pa.count_punctuation(text3)
Montaigne4=pa.count_punctuation(text4)
Montaigne5=pa.count_punctuation(text5)

print(Montaigne1)
#Here we plot our data

figure,axes=plt.subplots(2,3,figsize=[20,15])

axes[0][0].pie(Montaigne1.values(),labels=Montaigne1.keys())
axes[0][0].set_title("JoA")

axes[0][1].pie(Montaigne2.values(),labels=Montaigne2.keys())
axes[0][1].set_title("OfIdleness")

axes[0][2].pie(Montaigne3.values(),labels=Montaigne3.keys())
axes[0][2].set_title("OfLiars")

axes[1][0].pie(Montaigne4.values(),labels=Montaigne4.keys())
axes[1][0].set_title("OfSorrow")

axes[1][1].pie(Montaigne5.values(),labels=Montaigne5.keys())
axes[1][1].set_title("PD")

axes[1][2].remove()


figure.tight_layout(pad=4,w_pad=8,h_pad=2)


dan1=pa.count_punctuation(read_file("../input/DanBrown-DaA1.txt"))
dan2=pa.count_punctuation(read_file("../input/DanBrown-DaA2.txt"))
dan3=pa.count_punctuation(read_file("../input/DanBrown-DaA3-4.txt"))
dan4=pa.count_punctuation(read_file("../input/DanBrown-Origin1.txt"))
dan5=pa.count_punctuation(read_file("../input/DanBrown-OriginP.txt"))

figure,axes=plt.subplots(2,3,figsize=[15,10])

axes[0][0].pie(dan1.values(),labels=dan1.keys())
axes[0][0].set_title("AaD 1")

axes[0][1].pie(dan2.values(),labels=dan2.keys())
axes[0][1].set_title("AaD 2")

axes[0][2].pie(dan3.values(),labels=dan3.keys())
axes[0][2].set_title("AaD 2")

axes[1][0].pie(dan4.values(),labels=dan4.keys())
axes[1][0].set_title("Origin 1")

axes[1][1].pie(dan5.values(),labels=dan5.keys())
axes[1][1].set_title("Origin Prologue")

axes[1][2].remove()

figure.tight_layout(pad=4,w_pad=8,h_pad=2)


GRRM1=pa.count_punctuation(read_file("../input/GRRM-GOTPrologue.txt"))
GRRM2=pa.count_punctuation(read_file("../input/GRRM-GOTDany.txt"))
GRRM3=pa.count_punctuation(read_file("../input/GRRM-SOSPrologue.txt"))
GRRM4=pa.count_punctuation(read_file("../input/GRRM-SOSPrologue 2.txt"))
figure,axes=plt.subplots(2,2,figsize=[15,10])

axes[0][0].pie(GRRM1.values(),labels=GRRM1.keys())
axes[0][0].set_title("GoT Prologue")

axes[0][1].pie(GRRM2.values(),labels=GRRM2.keys())
axes[0][1].set_title("GOT Dany")


axes[1][0].pie(GRRM3.values(),labels=GRRM3.keys())
axes[1][0].set_title("SoS Prologue 1")

axes[1][1].pie(GRRM4.values(),labels=GRRM4.keys())
axes[1][1].set_title("Sos Prologue 2")


figure.tight_layout(pad=4,w_pad=8,h_pad=2)


MontaigneDF=pa.to_Frame(Montaigne1)
MontaigneDF=pd.concat([MontaigneDF,pa.to_Frame(Montaigne2)])
MontaigneDF=pd.concat([MontaigneDF,pa.to_Frame(Montaigne3)])
MontaigneDF=pd.concat([MontaigneDF,pa.to_Frame(Montaigne4)])
MontaigneDF=pd.concat([MontaigneDF,pa.to_Frame(Montaigne5)])

MontaigneDF["Name"]="Essays"



MontaigneDF

DanDf=pa.to_Frame(dan1)
DanDf=pd.concat([DanDf,pa.to_Frame(dan2)])
DanDf=pd.concat([DanDf,pa.to_Frame(dan3)])
DanDf=pd.concat([DanDf,pa.to_Frame(dan4)])
DanDf=pd.concat([DanDf,pa.to_Frame(dan5)])

DanDf["Name"]="Angels and Demons"
DanDf["Name"].iloc[3:]="Origin"

DanDf


GrrmDf=pa.to_Frame(GRRM1)
GrrmDf=pd.concat([GrrmDf,pa.to_Frame(GRRM2)])
GrrmDf=pd.concat([GrrmDf,pa.to_Frame(GRRM3)])
GrrmDf=pd.concat([GrrmDf,pa.to_Frame(GRRM4)])

GrrmDf["Name"]="GoT"
GrrmDf["Name"].iloc[2:]="SoS"


GrrmDf
TextsDf=pd.concat([MontaigneDF,DanDf,GrrmDf])
TextsDf

from sklearn.cross_validation import train_test_split
X=TextsDf.iloc[:,:-1]
Y=TextsDf.iloc[:,-1:]

OriginX=X
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X=ss.fit_transform(X)

x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size=0.33,random_state=0)


from sklearn.metrics import confusion_matrix #We will rate algorithms with confusion matrix

from sklearn.neighbors import KNeighborsClassifier #KNN algorithm
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)
y_predict=knn.predict(x_test)

print("KNN")
print(confusion_matrix(y_test,y_predict))
print("4 out of 5")
print("-------------------")

from sklearn.svm import SVC #Support Vector Classifier Algorithm
svc=SVC(kernel="poly",degree=3)
svc.fit(x_train,y_train)
y_predict=svc.predict(x_test)

print("SVC")
print(confusion_matrix(y_test,y_predict))
print("2 out of 5")

print("-------------------")

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=3)
rf.fit(x_train,y_train)
y_predict=rf.predict(x_test)

print("Random Forest")
print(confusion_matrix(y_test,y_predict))
print("3 out of 5")

print("-------------------")

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
y_predict=nb.predict(x_test)

print("Naive Bayes")
print(confusion_matrix(y_test,y_predict))
print("3 out of 5")

print("-------------------")

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(x_train,y_train)
y_predict=lr.predict(x_test)

print("Logistic Regression")
print(confusion_matrix(y_test,y_predict))
print("5 out of 5")

print("-------------------")
MontaigneNew=read_file("../input/OfDH.txt")
MontaigneNew=pa.count_punctuation(MontaigneNew)
MontaigneNew=pa.to_Frame(MontaigneNew)

ss=StandardScaler()
NewDataset=pd.concat([OriginX,MontaigneNew])
NewDataset=ss.fit_transform(NewDataset)
MontaigneNew=NewDataset[-1]
MontaigneNew=[MontaigneNew]

print(knn.predict(MontaigneNew))
print(lr.predict(MontaigneNew))