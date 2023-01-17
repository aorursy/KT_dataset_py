# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.pandas.set_option('display.max_columns',None)
chat = pd.read_csv("../input/whatsapp-chat/WhatsApp Chat with -gence.txt", header = None , error_bad_lines = False,encoding = 'utf8')

chat.head()
chat.drop([2, 3], axis = 1, inplace = True)

chat
chat.columns = ["Date", "Chat"]

chat
message = chat["Chat"].str.split("-", n= 1, expand = True)

message.columns = ["Time", "Message"]

message
chat = chat.drop("Chat", axis = 1)

chat["Time"] = message["Time"]

chat["Message"] = message["Message"]

chat.head(5)
message = chat["Message"].str.split(":", n= 1, expand = True)

message.columns = ["Name", "Message"]

message
chat = chat.drop("Message", axis = 1)

chat["Name"] = message["Name"]

chat["Message"] = message["Message"]

chat.head(10)
chat['Message'] = chat['Message'].str.lower()

chat.dropna(axis = 0, inplace = True)

chat.head()
chat
chat["Message"] = chat["Message"].str.replace(" <media omitted>", "Media_Shared")

chat["Message"] = chat["Message"].str.replace(" this message was deleted", "Message_Deleted")

chat.head(5)
print(chat["Name"].unique())
chat.reset_index(inplace = True)

chat.drop(['index'],axis=1,inplace=True)

chat["Name"] = chat["Name"].str.replace("Urban Pendu", "Yash Arora")

chat["Name"] = chat["Name"].str.replace("Abhishek tyagi", "Abhishek Tyagi")

chat["Name"] = chat["Name"].str.replace("Sahil Dite", "Sahil Kumar")

chat["Name"] = chat["Name"].str.replace("Sidhant Dite", "Siddhant Jha")

chat
msg_sent_per_member = chat["Name"].value_counts()

msg_sent_per_member
print(chat["Date"].value_counts().head(5))

print("Number of Days in which messgaes were exchanged : ", chat["Date"].value_counts().count())
media_shared = chat["Message"] == "Media_Shared"

media = chat[media_shared]

media["Name"].value_counts()
message_deleted = chat["Message"] == "Message_Deleted"

deleted = chat[message_deleted]

deleted["Name"].value_counts()