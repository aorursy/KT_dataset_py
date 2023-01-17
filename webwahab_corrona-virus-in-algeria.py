# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.svm import SVR



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

plt.rcParams.update({'font.size': 14})

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
confirmed_case=pd.read_csv("/kaggle/input/corona-virus-report/covid_19_clean_complete.csv")



confirmed_case.describe()



confirmed_case_selected= confirmed_case[["Province/State","Country/Region","Date","Confirmed","Deaths","Recovered"]] 



confirmed_case_selected_Algeria=confirmed_case_selected[confirmed_case_selected["Country/Region"] == "Algeria"]

confirmed_case_selected_France=confirmed_case_selected[confirmed_case_selected["Country/Region"] == "France"]

confirmed_case_selected_France=confirmed_case_selected_France.groupby(["Date"], as_index = False).sum().sort_values(by=["Confirmed"])

confirmed_case_selected_Germany=confirmed_case_selected[confirmed_case_selected["Country/Region"] == "Germany"]

confirmed_case_selected_SKorea=confirmed_case_selected[confirmed_case_selected["Country/Region"] == 'South Korea']

confirmed_case_selected_Italy=confirmed_case_selected[confirmed_case_selected["Country/Region"] == 'Italy']

confirmed_case_selected_China=confirmed_case_selected[confirmed_case_selected["Country/Region"] == 'China']

confirmed_case_selected_China=confirmed_case_selected_China.groupby(["Date"], as_index = False).sum().sort_values("Confirmed")

#confirmed_case_selected_China=confirmed_case_selected_China.groupby(["Country/Region","Date"], as_index = False).sum()

#confirmed_case_selected_France

x1 = np.linspace(1, confirmed_case_selected_Algeria["Confirmed"].shape[0], confirmed_case_selected_Algeria["Confirmed"].shape[0])

x2 = np.linspace(1, confirmed_case_selected_Germany["Confirmed"].shape[0], confirmed_case_selected_Germany["Confirmed"].shape[0])

x3 = np.linspace(1, confirmed_case_selected_SKorea["Confirmed"].shape[0], confirmed_case_selected_SKorea["Confirmed"].shape[0])

x4 = np.linspace(1, confirmed_case_selected_France["Confirmed"].shape[0], confirmed_case_selected_France["Confirmed"].shape[0])

x5 = np.linspace(1, confirmed_case_selected_Italy["Confirmed"].shape[0], confirmed_case_selected_Italy["Confirmed"].shape[0])

x6 = np.linspace(1, confirmed_case_selected_China["Confirmed"].shape[0], confirmed_case_selected_China["Confirmed"].shape[0])



confirmed_case_selected_Algeria["Deaths_Ratio"] = (confirmed_case_selected_Algeria["Deaths"]/confirmed_case_selected_Algeria["Confirmed"])*100

confirmed_case_selected_Algeria["Recovered_Ratio"] = (confirmed_case_selected_Algeria["Recovered"]/confirmed_case_selected_Algeria["Confirmed"])*100

confirmed_case_selected_Algeria["Deaths_Population_Ratio"] = (confirmed_case_selected_Algeria["Deaths"]/43851044)*100

confirmed_case_selected_Algeria["Day"] = x1

confirmed_case_selected_Algeria=confirmed_case_selected_Algeria.fillna(0)



confirmed_case_selected_France["Deaths_Ratio"] = (confirmed_case_selected_France["Deaths"]/confirmed_case_selected_France["Confirmed"])*100

confirmed_case_selected_France["Recovered_Ratio"] = (confirmed_case_selected_France["Recovered"]/confirmed_case_selected_France["Confirmed"])*100

confirmed_case_selected_France["Day"] = x4

confirmed_case_selected_France=confirmed_case_selected_France.fillna(0)



confirmed_case_selected_Germany["Deaths_Ratio"] = (confirmed_case_selected_Germany["Deaths"]/confirmed_case_selected_Germany["Confirmed"])*100

confirmed_case_selected_Germany["Recovered_Ratio"] = (confirmed_case_selected_Germany["Recovered"]/confirmed_case_selected_Germany["Confirmed"])*100



confirmed_case_selected_Germany=confirmed_case_selected_Germany.fillna(0)



confirmed_case_selected_Italy["Deaths_Ratio"] = (confirmed_case_selected_Italy["Deaths"]/confirmed_case_selected_Italy["Confirmed"])*100

confirmed_case_selected_Italy["Recovered_Ratio"] = (confirmed_case_selected_Italy["Recovered"]/confirmed_case_selected_Italy["Confirmed"])*100

confirmed_case_selected_Italy=confirmed_case_selected_Italy.fillna(0)



confirmed_case_selected_SKorea["Deaths_Ratio"] = (confirmed_case_selected_SKorea["Deaths"]/confirmed_case_selected_SKorea["Confirmed"])*100

confirmed_case_selected_SKorea["Recovered_Ratio"] = (confirmed_case_selected_SKorea["Recovered"]/confirmed_case_selected_SKorea["Confirmed"])*100

confirmed_case_selected_SKorea["Deaths_Population_Ratio"] = (confirmed_case_selected_SKorea["Deaths"]/51269185)*100

confirmed_case_selected_SKorea["Day"] = x3

confirmed_case_selected_SKorea=confirmed_case_selected_SKorea.fillna(0)



confirmed_case_selected_China["Deaths_Ratio"] = (confirmed_case_selected_China["Deaths"]/confirmed_case_selected_China["Confirmed"])*100

confirmed_case_selected_China["Recovered_Ratio"] = (confirmed_case_selected_China["Recovered"]/confirmed_case_selected_China["Confirmed"])*100

confirmed_case_selected_China=confirmed_case_selected_China.fillna(0)





print(confirmed_case_selected_Algeria)







fig, axes = plt.subplots(3,2) 

axes[0, 0].plot(x1, confirmed_case_selected_Algeria["Confirmed"], label='Algeria',color="red")



#axes.set_xlabel('x label')  # Add an x-label to the axes.

#axes.set_ylabel('y label')  # Add a y-label to the axes.

axes[0, 0].set_title("Algeria")  # Add a title to the axes.

axes[0, 0].legend()  # Add a legend







axes[0, 1].plot(x2, confirmed_case_selected_Germany["Confirmed"], label='Germany')

axes[0, 1].set_title("Germany")  # Add a title to the axes.

axes[0, 1].legend()  # Add a legend



axes[1, 0].plot(x3, confirmed_case_selected_SKorea["Confirmed"], label='South Korea',color="green")

axes[1, 0].set_title("South Korea")  # Add a title to the axes.

axes[1, 0].legend()  # Add a legend



axes[1, 1].plot(x4, confirmed_case_selected_France["Confirmed"], label='France',color="red")

axes[1, 1].set_title("France")  # Add a title to the axes.

axes[1, 1].legend()  # Add a legend



axes[2, 0].plot(x5, confirmed_case_selected_Italy["Confirmed"], label='Italy')

axes[2, 0].set_title("Italy")  # Add a title to the axes.

axes[2, 0].legend()  # Add a legend



axes[2, 1].plot(x6, confirmed_case_selected_China["Confirmed"], label='China',color="green")

axes[2, 1].set_title("China")  # Add a title to the axes.

axes[2, 1].legend()  # Add a legend



fig.set_size_inches(18.5, 10.5)

fig.tight_layout(pad=3.0)



for i in range (0,3):

    for j in range (0,2):

        axes[i,j].set_xlabel('Days')  # Add an x-label to the axes.

        axes[i,j].set_ylabel('Confirmed cases')  # Add a y-label to the axes.





fig1, axes1 = plt.subplots(2,2) 





axes1[0,0].plot(x1, confirmed_case_selected_Algeria["Deaths"], label='Algeria')

axes1[0,0].plot(x4, confirmed_case_selected_France["Deaths"], label='France')

#axes1[0,0].plot(x2, confirmed_case_selected_Germany["Deaths_Ratio"], label='Germany')

axes1[0,0].plot(x3, confirmed_case_selected_SKorea["Deaths"], label='South Korea')

axes1[0,0].plot(x6, confirmed_case_selected_China["Deaths"], label='China')

#axes1[0,0].plot(x5, confirmed_case_selected_Italy["Deaths_Ratio"], label='Italy')



#axes.set_xlabel('x label')  # Add an x-label to the axes.

#axes.set_ylabel('y label')  # Add a y-label to the axes.

axes1[0,0].set_xlabel('Days')  # Add an x-label to the axes.

axes1[0,0].set_ylabel('# of deaths')  # Add a y-label to the axes.

axes1[0,0].set_title("Deaths")  # Add a title to the axes.

axes1[0,0].legend()  # Add a legend



axes1[0,1].plot(x1, confirmed_case_selected_Algeria["Deaths_Ratio"], label='Algeria')

axes1[0,1].plot(x4, confirmed_case_selected_France["Deaths_Ratio"], label='France')

#axes1[0,0].plot(x2, confirmed_case_selected_Germany["Deaths_Ratio"], label='Germany')

axes1[0,1].plot(x3, confirmed_case_selected_SKorea["Deaths_Ratio"], label='South Korea')

axes1[0,1].plot(x6, confirmed_case_selected_China["Deaths_Ratio"], label='China')

#axes1[0,0].plot(x5, confirmed_case_selected_Italy["Deaths_Ratio"], label='Italy')



#axes.set_xlabel('x label')  # Add an x-label to the axes.

#axes.set_ylabel('y label')  # Add a y-label to the axes.

axes1[0,1].set_xlabel('Days')  # Add an x-label to the axes.

axes1[0,1].set_ylabel('%') 

axes1[0,1].set_title("Death Ratio")  # Add a title to the axes.

axes1[0,1].legend()  # Add a legend









axes1[1,0].plot(x1, confirmed_case_selected_Algeria["Deaths_Population_Ratio"], label='Algeria')

axes1[1,0].plot(x4, (confirmed_case_selected_France["Deaths"]/65270000)*100, label='France')

#axes1[1,0].plot(x2, (confirmed_case_selected_Germany["Deaths"]/83783942)*100, label='Germany')

axes1[1,0].plot(x3, confirmed_case_selected_SKorea["Deaths_Population_Ratio"], label='South Korea')

axes1[1,0].plot(x6, (confirmed_case_selected_China["Deaths"]/(1439323776))*100, label='China')

#axes1[1,0].plot(x5, (confirmed_case_selected_Italy["Deaths"]/60486683)*100 , label='Italy')

axes1[1,0].set_xlabel('Days')  # Add an x-label to the axes.

axes1[1,0].set_ylabel('%')

axes1[1,0].set_title("Death/Population")  # Add a title to the axes.

axes1[1,0].legend()  # Add a legend



axes1[1,1].plot(x1, confirmed_case_selected_Algeria["Recovered_Ratio"], label='Algeria')

axes1[1,1].plot(x4, confirmed_case_selected_France["Recovered_Ratio"], label='France')

#axes1[0,0].plot(x2, confirmed_case_selected_Germany["Deaths_Ratio"], label='Germany')

axes1[1,1].plot(x3, confirmed_case_selected_SKorea["Recovered_Ratio"], label='South Korea')

axes1[1,1].plot(x6, confirmed_case_selected_China["Recovered_Ratio"], label='China')

#axes1[0,0].plot(x5, confirmed_case_selected_Italy["Deaths_Ratio"], label='Italy')



#axes.set_xlabel('x label')  # Add an x-label to the axes.

#axes.set_ylabel('y label')  # Add a y-label to the axes.

axes1[1,1].set_xlabel('Days')  # Add an x-label to the axes.

axes1[1,1].set_ylabel("%")

axes1[1,1].set_title("Recovered Ratio")  # Add a title to the axes.

axes1[1,1].legend()  # Add a legend





fig1.set_size_inches(18.5, 10.5)



fig1.tight_layout(pad=3.0)
X= confirmed_case_selected_SKorea["Day"]

y= confirmed_case_selected_SKorea["Deaths_Population_Ratio"]

X=np.array(X).reshape(-1, 1)

y=np.array(y)



X_train=X[30:50]

X_test=X[50:57]



y_train=y[30:50]

y_test=y[50:57]





# y_train, y_test = train_test_split(X, y, test_size=0.2)



regressor = LinearRegression()  



regressor.fit(X_train, y_train) #training the algorithm



print (regressor.score(X_train,y_train))

print("Coefficient for X ", regressor.coef_)

#clf = SVR(C=0.5, epsilon=0.1)

#clf.fit(X_train, y_train)

y_pred = regressor.predict(X_test)



#y_pred = clf.predict(X_test)



df = pd.DataFrame({'Actual': y_test*10000000, 'Predicted': y_pred*10000000})

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



df

#confirmed_case_selected_SKorea



confirmed_case_selected_Algeria [["Day","Recovered_Ratio"]][50:]



x_incom_day= list(range(33, 63)) 

#np.linspace(1, confirmed_case_selected_Algeria["Confirmed"].shape[0], confirmed_case_selected_Algeria["Confirmed"].shape[0])

x_incom_day2=np.array(x_incom_day).reshape(-1, 1)



y_pred_incom_day = regressor.predict(x_incom_day2)





Death_radio_predection_pop_Algeria= pd.DataFrame({'Days': x_incom_day, 'Predicted': y_pred_incom_day})

Death_radio_predection_pop_Algeria['Days']= Death_radio_predection_pop_Algeria['Days'] +20



confirmed_case_selected_France["Deaths_Ratio"]





Death_radio_predection_Algeria=pd.DataFrame()

Death_radio_predection_Algeria["Day"]= confirmed_case_selected_France["Day"][29:]

Death_radio_predection_Algeria["Day"]= Death_radio_predection_Algeria["Day"]+22

Death_radio_predection_Algeria["Predicted"]= confirmed_case_selected_France["Deaths_Ratio"][29:]













#f = pd.DataFrame({'Actual': x_incom_day, 'Predicted': y_pred_incom_day *10000000})

#df 

#confirmed_case_selected_Algeria [["Day","Deaths_Population_Ratio"]][52:]

















fig2, axes2 = plt.subplots(3,2) 



axes2[0,0].plot(x1, confirmed_case_selected_Algeria["Deaths_Population_Ratio"], label='Algeria')

axes2[0,0].plot(Death_radio_predection_pop_Algeria['Days'], Death_radio_predection_pop_Algeria['Predicted'],linestyle='dashed' ,label='Algeria_predicted')

axes2[0,0].set_title("Predicted deaths/population Ratio ")  # Add a title to the axes.

axes2[0,0].legend()  # Add a legend



axes2[0,0].remove()



axes2[0,1].plot(x1, confirmed_case_selected_Algeria["Deaths_Ratio"], label='Algeria')

axes2[0,1].plot(Death_radio_predection_Algeria['Day'], Death_radio_predection_Algeria['Predicted'],linestyle='dashed',label='Algeria_predicted')

axes2[0,1].set_title("Predicted deaths_Ratio ")  # Add a title to the axes.

axes2[0,1].legend()  # Add a legend



axes2[0,1].remove()



axes2[1,0].plot(x1, confirmed_case_selected_Algeria["Deaths"], label='Algeria')

axes2[1,0].plot(Death_radio_predection_pop_Algeria['Days'], (Death_radio_predection_pop_Algeria['Predicted']*43851044)/100, label='Algeria_predicted')

axes2[1,0].set_xlabel('Days')

axes2[1,0].set_ylabel('# of Deaths')

axes2[1,0].set_title("Predicted death cases")  # Add a title to the axes.

axes2[1,0].legend()  # Add a legend



Death_radio_predection_Algeria = Death_radio_predection_Algeria[1:29]

Death_radio_predection_pop_Algeria = Death_radio_predection_pop_Algeria[0:28]

Death_radio_predection_Algeria["Predicted"] =Death_radio_predection_Algeria["Predicted"]/100

Confirmed_predicted= pd.DataFrame()

Confirmed_predicted["Days"]= Death_radio_predection_pop_Algeria["Days"]

Confirmed_predicted["predicted_death"]=(Death_radio_predection_pop_Algeria["Predicted"]*43851044)/100

Confirmed_predicted["predicted_radio_pop"]= list(Death_radio_predection_Algeria["Predicted"])

Confirmed_predicted["predicted_confirmed"]= Confirmed_predicted["predicted_death"]/Confirmed_predicted["predicted_radio_pop"]



axes2[1,1].plot(x1, confirmed_case_selected_Algeria["Confirmed"], label='Algeria')

axes2[1,1].plot(Confirmed_predicted["Days"], Confirmed_predicted["predicted_confirmed"], linestyle='dashed',label='Algeria_predicted')

#axes2[1,1].plot(x2, confirmed_case_selected_Germany["Confirmed"], label='Germany')

#axes2[1,1].plot(x3, confirmed_case_selected_SKorea["Confirmed"], label='South Korea')

#axes2[1,1].plot(x4, confirmed_case_selected_France["Confirmed"], label='France')

#axes2[1,1].plot(x5, confirmed_case_selected_Italy["Confirmed"], label='Italy')

#axes2[1,1].plot(x6, confirmed_case_selected_China["Confirmed"], label='China')

axes2[1,1].set_xlabel('Days')

axes2[1,1].set_ylabel('# of Confirmed Cases')

axes2[1,1].set_title("Predicted confirmed case")  # Add a title to the axes.

axes2[1,1].legend()  # Add a legend





axes2[2,0].plot(x1, confirmed_case_selected_Algeria["Deaths"], label='Algeria')

axes2[2,0].plot(Death_radio_predection_pop_Algeria['Days'], (Death_radio_predection_pop_Algeria['Predicted']*43851044)/100, label='Algeria_predicted')

axes2[2,0].plot(x2, confirmed_case_selected_Germany["Deaths"], label='Germany')

axes2[2,0].plot(x3, confirmed_case_selected_SKorea["Deaths"], label='South Korea')

axes2[2,0].plot(x4, confirmed_case_selected_France["Deaths"], label='France')

axes2[2,0].plot(x5, confirmed_case_selected_Italy["Deaths"], label='Italy')

axes2[2,0].plot(x6, confirmed_case_selected_China["Deaths"], label='China')

axes2[2,0].set_xlabel('Days')

axes2[2,0].set_ylabel('# of Deaths')

axes2[2,0].set_title("Compare predicted death cases")  # Add a title to the axes.

axes2[2,0].legend()  # Add a legend





axes2[2,1].plot(x1, confirmed_case_selected_Algeria["Confirmed"], label='Algeria')

axes2[2,1].plot(Confirmed_predicted["Days"], Confirmed_predicted["predicted_confirmed"], linestyle='dashed',label='Algeria_predicted')

axes2[2,1].plot(x2, confirmed_case_selected_Germany["Confirmed"], label='Germany')

axes2[2,1].plot(x3, confirmed_case_selected_SKorea["Confirmed"], label='South Korea')

axes2[2,1].plot(x4, confirmed_case_selected_France["Confirmed"], label='France')

axes2[2,1].plot(x5, confirmed_case_selected_Italy["Confirmed"], label='Italy')

axes2[2,1].plot(x6, confirmed_case_selected_China["Confirmed"], label='China')



axes2[2,1].set_xlabel('Days')

axes2[2,1].set_ylabel('# of Confirmed Cases')  # Add a y-label to the axes.



axes2[2,1].set_title("Compare predicted confirmed case")  # Add a title to the axes.

axes2[2,1].legend()  # Add a legend















fig2.set_size_inches(18.5, 15.5)



fig2.tight_layout(pad=3.0)







#print(Death_radio_predection_Algeria["Predicted"])

#print(confirmed_case_selected_Algeria)

print(Confirmed_predicted)





#x_incom_day

#y_pred_incom_day




