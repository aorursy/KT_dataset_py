
#Import necessary packages
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import os
print(os.listdir("../input"))

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
train_spl = pd.read_csv("../input/train_sample.csv")
train_spl.info()
#Create subset for download apps and NOT download apps
dld_train_spl = train_spl[train_spl['is_attributed']==1]
dld_train_spl.info()

no_dld_train_spl = train_spl[train_spl['is_attributed']==0]
no_dld_train_spl.info()
ip_no_dld = no_dld_train_spl["ip"].value_counts()
ip_no_dld[:10]
#Plot the ip addresses which are not download apps
sns.countplot(x = "ip",  data = no_dld_train_spl,\
             order = ip_no_dld[:10].index).set(\
            xlabel = "ip address not download the app")
#Extract the top 10 ip addresses(not download the app) from the download dataset
ip_in_dld = dld_train_spl.loc[dld_train_spl['ip'].isin(ip_no_dld[:10].index)]
ip_in_dld.shape

#Group by ip
groupby_ip_dld = ip_in_dld.groupby("ip").size()
groupby_ip_dld[:10]
#In train sample set, extract the ip addresses whcih not download the app
#In this way, we can find the total click amount
ip_no_dld_in_all = train_spl.loc[train_spl['ip'].isin(ip_no_dld[:10].index)]
ip_no_dld_in_all.shape

#Group by ip
groupby_no_dld_ip=ip_no_dld_in_all.groupby("ip").size()
groupby_no_dld_ip

#Plot the total click amount by ip addresses
sns.countplot(x = 'ip', data = ip_no_dld_in_all,\
              order = ip_no_dld[:10].index).set(\
            xlabel = "ip address click amount")
#Merge the download times and click times based on same ip addresses
click_dld_ = pd.merge(groupby_no_dld_ip.reset_index(),ip_no_dld[:10].reset_index(),\
                      left_on = "ip", right_on = "index").iloc[:,[0,1,3]]
click_dld_.columns = ['ip','click times', 'not download times']
click_dld_['download times'] = click_dld_['click times'] - click_dld_['not download times']

#Calculate the download rate
click_dld_['download rate'] = click_dld_['download times']/ click_dld_['click times']
click_dld_

#Average download rate for the 10 ip addresses
print("Average download rate is: " , click_dld_["download rate"].mean())
#Top 10 ip addresses which download the apps
ip_dld = dld_train_spl["ip"].value_counts()
ip_dld[:10]

ip_dld_in_all = train_spl.loc[train_spl['ip'].isin(ip_dld[:10].index)]

sns.countplot(x = 'ip', data = ip_dld_in_all,\
              order = ip_dld[:10].index).set(\
            xlabel = "ip address click amount")
#In train sample set, extract the ip addresses whcih download the app
ip_dld_in_all = train_spl.loc[train_spl['ip'].isin(ip_dld[2:12].index)]

#Group by ip
groupby_dld_ip=ip_dld_in_all.groupby("ip").size()
groupby_dld_ip

#Plot the total click amount by ip addresses
sns.countplot(x = 'ip', data = ip_dld_in_all,\
              order = ip_dld[2:12].index).set(\
            xlabel = "ip address click amount")
#Merge the download times and click times based on same ip addresses
click_dld = pd.merge(groupby_dld_ip.reset_index(),ip_dld[2:12].reset_index(),\
                      left_on = "ip", right_on = "index").iloc[:,[0,1,3]]
click_dld.columns = ['ip','click times', 'download times']

#Calculate the download rate
click_dld['download rate'] = click_dld['download times']/ click_dld['click times']
click_dld

#Average download rate for the 10 ip addresses
print("Average download rate is: " , click_dld["download rate"].mean())
