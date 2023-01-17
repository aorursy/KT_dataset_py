import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
xlsx_file = pd.ExcelFile('../input/Faults.xlsx')
xlsx_file
xlsx_file.sheet_names
june_df = xlsx_file.parse('June 2018')
july_df = xlsx_file.parse('July 2018')
june_df.head()
whole_df = june_df.append(july_df)
whole_df = whole_df[["Date Fault Occurred", "Car", "Fault Description", "Level 1 Classification (MAXIMO)", "Fault", "Work Order", "Status", "Repeated Fault", "Comments" ]]

whole_df.head()
fault2_rows = whole_df.loc[whole_df["Repeated Fault"] == "Yes"]
fault2_rows.head()

fault2_faultID = fault2_rows[fault2_rows["Comments"].str.contains("Repeat of fault")]
fault2_faultWO = fault2_rows[fault2_rows["Comments"].str.contains("Repeat of WO")]
fault2_faultID.head()
fault1_ID = []

for i in range(len(fault2_faultID)):
    comment = fault2_faultID["Comments"].iloc[i]
    comment_array = comment.split(" ")
    index = comment_array.index("fault")
    ID = comment_array[index + 1][0:6]
    fault1_ID.append(ID)
    
fault2_ID = fault2_faultID["Fault"]
pairs1 = pd.DataFrame({"Fault 1": fault1_ID, "Fault 2" : fault2_ID })
pairs1
fault1_ID2 = []
fault2_ID2 = fault2_faultWO["Fault"]

for i in range(len(fault2_faultWO)):
    comment = fault2_faultWO["Comments"].iloc[i]
    comment_array = comment.split(" ")
    index = comment_array.index("WO")
    WO = comment_array[index + 1][0:7]
    WO = float(WO)
    f1 = whole_df.loc[whole_df["Work Order"] == WO]
    fault1_id = f1["Fault"]
    
    if fault1_id.empty:
        fault1_ID2.append(np.nan)
    else:
        fault1_ID2.append(fault1_id.item())
        
pairs2 = pd.DataFrame({"Fault 1": fault1_ID2, "Fault 2" : fault2_ID2 })
pairs2
all_pairs = pairs1.append(pairs2)
all_pairs = all_pairs.dropna(axis=0)
all_pairs
for i in range(len(all_pairs)):
    ID_1 = all_pairs["Fault 1"].iloc[i]
    ID_2 = all_pairs["Fault 2"].iloc[i]
    fault1 = whole_df.loc[whole_df["Fault"] == int(ID_1)].drop(["Work Order", "Repeated Fault", "Comments"], axis=1)
    fault2 = whole_df.loc[whole_df["Fault"] == ID_2].drop(["Work Order", "Repeated Fault", "Comments"], axis=1)
    
    if not fault1.empty and not fault2.empty:
        
        print ("Fault 1: ")
        print("Date Fault Occurred : ", fault1["Date Fault Occurred"].iloc[0])
        print("Car : ", fault1["Car"].iloc[0])
        print("Fault Description : ", fault1["Fault Description"].iloc[0])
        print("Level 1 Classification (MAXIMO) : ", fault1["Level 1 Classification (MAXIMO)"].iloc[0])
        print("Fault ID : ", fault1["Fault"].iloc[0])
        print("Status : ", fault1["Status"].iloc[0])
        print(" ")
    
        print ("Fault 2: ")
        print("Date Fault Occurred : ", fault2["Date Fault Occurred"].iloc[0])
        print("Car : ", fault2["Car"].iloc[0])
        print("Fault Description : ", fault2["Fault Description"].iloc[0])
        print("Level 1 Classification (MAXIMO) : ", fault2["Level 1 Classification (MAXIMO)"].iloc[0])
        print("Fault ID : ", fault2["Fault"].iloc[0])
        print("Status : ", fault2["Status"].iloc[0])
        print(" ")
        print(" ")
    
    else: 
        print(" ")
        print ("Fault ", ID_1, " cannot be found in dataset")
        print(" ")
        print(" ")
