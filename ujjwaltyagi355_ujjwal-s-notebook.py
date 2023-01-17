

# -*- coding: utf-8 -*-

"""

Created on Tue Mar 17 20:07:03 2020



@author: ujjwal

"""







import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

new_data=pd.read_csv("C:/Users/ujjwal/Desktop/train.csv")

test_data=pd.read_csv("C:/Users/ujjwal/Desktop/test.csv")

new_data['Image Resolution'].fillna(0,inplace=True)

new_data['Image Resolution'].value_counts(dropna=False)

new_data['Image Resolution'] = new_data['Image Resolution'].apply(lambda x: float(str(x)[:4]))

#print(new_data['Image Resolution'].head)

new_data['Capacity'].fillna(3500,inplace=True)

new_data['Capacity'].value_counts(dropna=False)

new_data['Capacity'] = new_data['Capacity'].apply(lambda x: int(str(x)[:4]))

new_data['Internal Memory'].fillna(35,inplace=True)

#new_data['Internal Memory'].value_counts(dropna=False)

new_data['Internal Memory'] = new_data['Internal Memory'].apply(lambda x: int(str(x)[:2]))

new_data['RAM'].fillna(5,inplace=True)

new_data['RAM'].value_counts(dropna=False)

new_data['RAM'].replace(to_replace="512 MB",value="0",inplace=True)

new_data['RAM'].replace(to_replace="16 MB",value="0",inplace=True)

new_data['RAM'].replace(to_replace="64 MB",value="0",inplace=True)

new_data['RAM'].replace(to_replace="8 MB",value="0",inplace=True)

new_data['RAM'].replace(to_replace="1.5 GB Good â–¾RAM compared to7 - 11 K PhonesAll PhonesThis Device1.5 GBAverage in group2 GBBest in this group6 GBGroup: 7 - 11 K Phones Based on specs, benchmarks & expert ratingsSee Phones with Highest RAM in 7 - 11 K",value="1",inplace=True)

new_data['RAM'].replace(to_replace="32 MB",value="0",inplace=True)

new_data['RAM'].replace(to_replace="512 MB Below Average â–¾RAM compared to11 - 17 K PhonesAll PhonesThis Device512 MBAverage in group3 GBBest in this group6 GBGroup: 11 - 17 K Phones Based on specs, benchmarks & expert ratingsSee Phones with Highest RAM in 11 - 17 K",value="0",inplace=True)

new_data['RAM'].replace(to_replace="512 MB Below Average â–¾RAM compared to2 - 4 K PhonesAll PhonesThis Device512 MBAverage in group512 MBBest in this group2 GBGroup: 2 - 4 K Phones Based on specs, benchmarks & expert ratingsSee Phones with Highest RAM in 2 - 4 K",value="0",inplace=True)

new_data['RAM'].replace(to_replace="1.5 GB Good â–¾RAM compared to15 - 23 K PhonesAll PhonesThis Device1.5 GBAverage in group3 GBBest in this group8 GBGroup: 15 - 23 K Phones Based on specs, benchmarks & expert ratingsSee Phones with Highest RAM in 15 - 23 K",value="1",inplace=True)

new_data['RAM'].replace(to_replace="1.5 GB Good â–¾RAM compared to6 - 8 K PhonesAll PhonesThis Device1.5 GBAverage in group2 GBBest in this group4 GBGroup: 6 - 8 K Phones Based on specs, benchmarks & expert ratingsSee Phones with Highest RAM in 6 - 8 K",value="1",inplace=True)





new_data['RAM'] = new_data['RAM'].apply(lambda x: int(str(x)[:1])*10)

#print(new_data['RAM'].value_counts(),'\n')

new_data['Screen Size'].fillna(6,inplace=True)

new_data['Screen Size'].value_counts(dropna=False)

new_data['Screen Size'].replace(to_replace="6 inches (15.24 cm)",value="6.0",inplace=True)

new_data['Screen Size'].replace(to_replace="5 inches (12.7 cm)",value="5.0",inplace=True)

new_data['Screen Size'].replace(to_replace="2 inches (5.08 cm)",value="2.0",inplace=True)

new_data['Screen Size'] = new_data['Screen Size'].apply(lambda x: float(str(x)[:3])*10)

#print(new_data["Screen Size"].value_counts(),'\n')

new_data['Screen Resolution'].fillna(0,inplace=True)

new_data['Screen Resolution'].value_counts(dropna=False)

new_data['Screen Resolution'].replace(to_replace="Full HD (1080 x 1920 pixels)",value="1080",inplace=True)

new_data['Screen Resolution'].replace(to_replace="HD (720 x 1280 pixels)",value="720",inplace=True)

new_data['Screen Resolution'].replace(to_replace="Full HD (1440 x 1440 pixels)",value="1440",inplace=True)

new_data['Screen Resolution'] = new_data['Screen Resolution'].apply(lambda x: float(str(x)[:4]))

#print(new_data["Screen Resolution"].value_counts(),'\n')

new_data['Thickness'].fillna(8.0,inplace=True)

new_data['Thickness'].replace(to_replace="8 mm Very Good â–¾Thickness compared to10 - 14 K PhonesAll PhonesThis Device8 mmAverage in group8.3 mmBest in this group5.5 mmGroup: 10 - 14 K Phones Based on specs, benchmarks & expert ratingsSee Slimmest Phones in 10 - 14 K",value="8.0",inplace=True)

new_data['Thickness'].replace(to_replace="8 mm Very Good â–¾Thickness compared to8 - 12 K PhonesAll PhonesThis Device8",value="8.0",inplace=True)

new_data['Thickness'].replace(to_replace="8 mm Very Good â–¾Thickness compared to11 - 17 K PhonesAll PhonesThis Device8",value="8.0",inplace=True)

new_data['Thickness'].replace(to_replace="8 mm Very Good â–¾Thickness compared to11 - 17 K PhonesAll PhonesThis Device8 mmAverage in group8.2",value="8.0",inplace=True)

new_data['Thickness'].replace(to_replace="9 mm Good â–¾Thickness compared to4 - 6 K PhonesAll PhonesThis Device9 ",value="9.0",inplace=True)

new_data['Thickness'].replace(to_replace="8 mm Very Good â–¾Thickness compared to7 - 11 K PhonesAll PhonesThis Device8 mmAverage in group8.5 mmBest in this group5.5 mmGroup: 7 - 11 K Phones Based on specs, benchmarks & expert ratingsSee Slimmest Phones in 7 - 11 K",value="8.0",inplace=True)

new_data['Thickness'] = new_data['Thickness'].apply(lambda x: float(str(x)[:2]))



new_data['Thickness'].value_counts(dropna=False)

test_data['Image Resolution'].fillna(0,inplace=True)

test_data['Image Resolution'].value_counts(dropna=False)

test_data['Image Resolution'] = new_data['Image Resolution'].apply(lambda x: float(str(x)[:4]))

#print(new_data['Image Resolution'].head)

test_data['Capacity'].fillna(3500,inplace=True)

test_data['Capacity'].value_counts(dropna=False)

test_data['Capacity'] = new_data['Capacity'].apply(lambda x: int(str(x)[:4]))

test_data['Internal Memory'].fillna(35,inplace=True)

#new_data['Internal Memory'].value_counts(dropna=False)

test_data['Internal Memory'] = new_data['Internal Memory'].apply(lambda x: int(str(x)[:2]))

test_data['RAM'].fillna(5,inplace=True)

test_data['RAM'].value_counts(dropna=False)

test_data['RAM'].replace(to_replace="512 MB",value="0",inplace=True)

test_data['RAM'].replace(to_replace="16 MB",value="0",inplace=True)

test_data['RAM'].replace(to_replace="64 MB",value="0",inplace=True)

test_data['RAM'].replace(to_replace="8 MB",value="0",inplace=True)

test_data['RAM'].replace(to_replace="1.5 GB Good â–¾RAM compared to7 - 11 K PhonesAll PhonesThis Device1.5 GBAverage in group2 GBBest in this group6 GBGroup: 7 - 11 K Phones Based on specs, benchmarks & expert ratingsSee Phones with Highest RAM in 7 - 11 K",value="1",inplace=True)

test_data['RAM'].replace(to_replace="32 MB",value="0",inplace=True)

test_data['RAM'].replace(to_replace="512 MB Below Average â–¾RAM compared to11 - 17 K PhonesAll PhonesThis Device512 MBAverage in group3 GBBest in this group6 GBGroup: 11 - 17 K Phones Based on specs, benchmarks & expert ratingsSee Phones with Highest RAM in 11 - 17 K",value="0",inplace=True)

test_data['RAM'].replace(to_replace="512 MB Below Average â–¾RAM compared to2 - 4 K PhonesAll PhonesThis Device512 MBAverage in group512 MBBest in this group2 GBGroup: 2 - 4 K Phones Based on specs, benchmarks & expert ratingsSee Phones with Highest RAM in 2 - 4 K",value="0",inplace=True)

test_data['RAM'].replace(to_replace="1.5 GB Good â–¾RAM compared to15 - 23 K PhonesAll PhonesThis Device1.5 GBAverage in group3 GBBest in this group8 GBGroup: 15 - 23 K Phones Based on specs, benchmarks & expert ratingsSee Phones with Highest RAM in 15 - 23 K",value="1",inplace=True)

test_data['RAM'].replace(to_replace="1.5 GB Good â–¾RAM compared to6 - 8 K PhonesAll PhonesThis Device1.5 GBAverage in group2 GBBest in this group4 GBGroup: 6 - 8 K Phones Based on specs, benchmarks & expert ratingsSee Phones with Highest RAM in 6 - 8 K",value="1",inplace=True)





test_data['RAM'] = new_data['RAM'].apply(lambda x: int(str(x)[:1])*10)

#print(new_data['RAM'].value_counts(),'\n')

test_data['Screen Size'].fillna(6,inplace=True)

test_data['Screen Size'].value_counts(dropna=False)

test_data['Screen Size'].replace(to_replace="6 inches (15.24 cm)",value="6.0",inplace=True)

test_data['Screen Size'].replace(to_replace="5 inches (12.7 cm)",value="5.0",inplace=True)

test_data['Screen Size'].replace(to_replace="2 inches (5.08 cm)",value="2.0",inplace=True)

test_data['Screen Size'] = new_data['Screen Size'].apply(lambda x: float(str(x)[:3])*10)

#print(new_data["Screen Size"].value_counts(),'\n')

test_data['Screen Resolution'].fillna(0,inplace=True)

test_data['Screen Resolution'].value_counts(dropna=False)

test_data['Screen Resolution'].replace(to_replace="Full HD (1080 x 1920 pixels)",value="1080",inplace=True)

test_data['Screen Resolution'].replace(to_replace="HD (720 x 1280 pixels)",value="720",inplace=True)

test_data['Screen Resolution'].replace(to_replace="Full HD (1440 x 1440 pixels)",value="1440",inplace=True)

test_data['Screen Resolution'] = new_data['Screen Resolution'].apply(lambda x: float(str(x)[:4]))

#print(new_data["Screen Resolution"].value_counts(),'\n')

test_data['Thickness'].fillna(8.0,inplace=True)

test_data['Thickness'].replace(to_replace="8 mm Very Good â–¾Thickness compared to10 - 14 K PhonesAll PhonesThis Device8 mmAverage in group8.3 mmBest in this group5.5 mmGroup: 10 - 14 K Phones Based on specs, benchmarks & expert ratingsSee Slimmest Phones in 10 - 14 K",value="8.0",inplace=True)

test_data['Thickness'].replace(to_replace="8 mm Very Good â–¾Thickness compared to8 - 12 K PhonesAll PhonesThis Device8",value="8.0",inplace=True)

test_data['Thickness'].replace(to_replace="8 mm Very Good â–¾Thickness compared to11 - 17 K PhonesAll PhonesThis Device8",value="8.0",inplace=True)

test_data['Thickness'].replace(to_replace="8 mm Very Good â–¾Thickness compared to11 - 17 K PhonesAll PhonesThis Device8 mmAverage in group8.2",value="8.0",inplace=True)

test_data['Thickness'].replace(to_replace="9 mm Good â–¾Thickness compared to4 - 6 K PhonesAll PhonesThis Device9 ",value="9.0",inplace=True)

test_data['Thickness'].replace(to_replace="8 mm Very Good â–¾Thickness compared to7 - 11 K PhonesAll PhonesThis Device8 mmAverage in group8.5 mmBest in this group5.5 mmGroup: 7 - 11 K Phones Based on specs, benchmarks & expert ratingsSee Slimmest Phones in 7 - 11 K",value="8.0",inplace=True)

test_data['Thickness'] = new_data['Thickness'].apply(lambda x: float(str(x)[:2]))

new_data['Screen to Body Ratio (calculated)'].fillna(80,inplace=True)

new_data['Screen to Body Ratio (calculated)'] = new_data['Screen to Body Ratio (calculated)'].apply(lambda x: float(str(x)[:2]))

test_data['Screen to Body Ratio (calculated)'].fillna(80,inplace=True)

test_data['Screen to Body Ratio (calculated)'] = test_data['Screen to Body Ratio (calculated)'].apply(lambda x: float(str(x)[:2]))



test_data['Thickness'].value_counts(dropna=False)

z=new_data[['Rating']].values

count=0       

#print(z[1])

x=[]

#print(z.shape)

for i in z:

    if i>4.0 or i==4.0:

        x.append(1)

    else:

        x.append(0)

y=np.array(x).reshape(355,1)

for i in y:

    if i==0:

        count=count+1

print(count)





filtered_data=new_data[['Screen to Body Ratio (calculated)','Internal Memory','Screen Size','RAM','Capacity']]

filtered_test_data=test_data[['Screen to Body Ratio (calculated)','Internal Memory','Screen Size','RAM','Capacity']]





Threshold=0.3015

class NeuronLevel():

    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):

        self.new_weights = np.random.random((number_of_inputs_per_neuron, number_of_neurons))

        

class NeuralNetwork():

    def __init__(self,layer1,layer2,layer3,features,predictor,test_features):

        self.layer1=layer1

        self.layer2=layer2

        self.layer3=layer3

        self.features=features

        self.predictor=predictor

        self.test_features=test_features

        

    def Scaling(self,features):

        scaled_features=StandardScaler().fit_transform(features)

        

        #print(scaled_features[1])

        return scaled_features

    

    def Sig(self, hypothesis):

        activ=(1/(1+np.exp(-(hypothesis))))

        #activ= .5 * (1 + np.tanh(.5 * hypothesis))

        return activ

    def Activation(self,x):

        new=[]

        for i in x:

            activ= 1 if i>=Threshold else 0

            new.append(activ)    

        return new  

        

    def Hypothesis(self, scaled_features):

        hypo1=np.matmul(scaled_features,self.layer1.new_weights)

        output1=self.Sig(hypo1)

        hypo2=np.dot(output1,self.layer2.new_weights)

        output2=self.Sig(hypo2)

        hypo3=np.dot(output2,self.layer3.new_weights)

        output3=self.Sig(hypo3)

        return output1,output2,output3

    def __Devsig(self,x):

        return x*(1-x)

    def Fit(self,epochs,lr):

        scaled_features=self.Scaling(self.features)

        out_hist=[]

        accuracy=[]

        max_accuracy=0

        count=0

        for i in range(epochs):

            for x,y in zip(scaled_features,self.predictor):

                x=x.reshape(1,5)

                y=y.reshape(1,1)

                output1,output2,output3=self.Hypothesis(x)

                value=self.Activation(output3)

                out_hist.append(value)

                error_layer3 = (y-value)

                delta_layer3=np.dot(output2.T,error_layer3)

                err_layer2=np.dot(error_layer3,self.layer3.new_weights.T)

                error_layer2=err_layer2*self.__Devsig(output2)  

                delta_layer2=np.dot(output1.T,error_layer2)

                err_layer1=np.dot(error_layer2,self.layer2.new_weights.T)

                error_layer1=err_layer1*self.__Devsig(output1)

                delta_layer1=np.dot(x.T,error_layer1)

                self.layer1.new_weights+=delta_layer1

                self.layer2.new_weights+=delta_layer2

                self.layer3.new_weights+=delta_layer3   

            #print(self.layer1.new_weights)

            self.scaled_test_features=self.Scaling(self.test_features)

            count=count+1

            accuracy.append(accuracy_score(out_hist,self.predictor))

            if(accuracy[i]>max_accuracy):

                max_accuracy=accuracy[i]

                

                print(max_accuracy)

                

                chptw1=self.layer1.new_weights

                

                chptw2=self.layer2.new_weights

                chptw3=self.layer3.new_weights

                

                hypo1=np.matmul(self.scaled_test_features,chptw1)

                output1=self.Sig(hypo1)

        

                hypo2=np.dot(output1,chptw2)

      

                output2=self.Sig(hypo2)

    

                hypo3=np.dot(output2,chptw3)

                output3=self.Sig(hypo3)

                #rint("#############################################")

                #print(output3)

                #print("#############################################")

                value1=self.Activation(output3)

                #print("##################################")

                #print(value1)

                #print("##################################")

            out_hist.clear()

        print(max_accuracy)

        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        #print(value1)

        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        final_df = pd.DataFrame({'PhoneId': test_data['PhoneId'], 'Class': value1})

        final_df.to_csv('C:/Users/ujjwal/Desktop/submission78.csv', header=True, index=False)

        return [max_accuracy]    

        

layer1=NeuronLevel(5,5)

layer2=NeuronLevel(2,5)

layer3=NeuronLevel(1,2)

network=NeuralNetwork(layer1,layer2,layer3,filtered_data,y,filtered_test_data)

np.seterr(divide='raise')

final=network.Fit(5000,0.6)