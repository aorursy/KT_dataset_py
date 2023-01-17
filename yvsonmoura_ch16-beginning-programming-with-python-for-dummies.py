## Page 309

# Creating a simple library
class FormatData:
    def __init__(self, Name="", Age=0, Married=False):
        self.Name = Name
        self.Age = Age
        self.Married = Married
    def __str__(self):
        OutString = "'{0}', {1}, {2}".format(
            self.Name,
            self.Age,
            self.Married)
        return OutString

# Saving a .py file
%save FormatLib.py 1
# Importing a library and using it

import FormatLib

NewData = [FormatData("George", 65, True),
           FormatData("Sally", 47, False),
           FormatData("Doug", 52, True)]

for Entry in NewData:
    print(Entry)
## Pages 311/312

# Creating a File
import csv

class FormatData2:
    def __init__(self, Name="", Age=0, Married=False):
        self.Name = Name
        self.Age = Age
        self.Married = Married
    
    def __str__(self):
        OutString = "'{0}', {1}, {2}".format(
            self.Name,
            self.Age,
            self.Married)
        return OutString

    def SaveData(Filename = "", DataList = []):
        with open(Filename, "w", newline='\n') as csvfile: # "w" means write mode
            DataWriter = csv.writer(csvfile, delimiter='\n',quotechar=" ", quoting=csv.QUOTE_NONNUMERIC)
            DataWriter.writerow(DataList)
            csvfile.close()
            print("Data saved!")

NewData = [FormatData2("George", 65, True),
           FormatData2("Sally", 47, False),
           FormatData2("Doug", 52, True)]

FormatData2.SaveData("TestFile.csv", NewData)
## Page 314/315/316

# Reading File Content

import csv

class FormatData3:
    def __init__(self, Name="", Age=0, Married=False):
        self.Name = Name
        self.Age = Age
        self.Married = Married
    
    def __str__(self):
        OutString = "'{0}', {1}, {2}".format(
            self.Name,
            self.Age,
            self.Married)
        return OutString
    
    def SaveData(Filename = "", DataList = []):
        with open(Filename, "w", newline='\n') as csvfile:
            DataWriter = csv.writer(csvfile,delimiter='\n',quotechar=" ",quoting=csv.QUOTE_NONNUMERIC)
            DataWriter.writerow(DataList)
            csvfile.close()
            print("Data saved!")

    def ReadData(Filename = ""):
        with open(Filename, "r", newline='\n') as csvfile:
            DataReader = csv.reader(csvfile, delimiter="\n", quotechar=" ", quoting=csv.QUOTE_NONNUMERIC)
            Output = []
        
            for Item in DataReader:
                Output.append(Item[0])
            
            csvfile.close()
            print("Data read!")
            
        return Output
    
NewData3 = [FormatData3("George", 65, True),
           FormatData3("Sally", 47, False),
           FormatData3("Doug", 52, True)]

FormatData3.SaveData("TestCSV.csv", NewData3)
FormatData3.ReadData("TestCSV.csv")
## Page 317/318/319

# Updating File Content

import os.path

if not os.path.isfile("Testfile.csv"):
    print("Please run the CreateFile.py example!")
    quit()

NewData = FormatData3.ReadData("TestFile.csv")

for Entry in NewData:
    print(Entry)

print("\r\nAdding a record for Harry.")
NewRecord = "'Harry', 23, False"
NewData.append(NewRecord)

for Entry in NewData:
    print(Entry)

print("\r\nRemoving Doug's record.")
Location = NewData.index("'Doug', 52, True")
Record = NewData[Location]
NewData.remove(Record)

for Entry in NewData:
    print(Entry)

print("\r\nModifying Sally's record.")
Location = NewData.index("'Sally', 47, False")
Record = NewData[Location]
Split = Record.split(",")

NewRecord = FormatData3(Split[0].replace("'", ""),
                        int(Split[1]),
                        bool(Split[2]))
NewRecord.Married = True
NewRecord.Age = 48
NewData.append(NewRecord.__str__())
NewData.remove(Record)

for Entry in NewData:
    print(Entry)

FormatData3.SaveData("ChangedFile.csv", NewData)

## Page 321

# Deleting a File
import os

# os.rmdir(): Removes the specified directory. The directory must be empty or Python will display an exception message

# shutil.rmtree(): Removes the specified directory, all subdirectories, and all files. This function is especially dangerous because it removes everything
# without checking (Python assumes that you know what you’re doing). As a result, you can easily lose data using this function.

os.remove("ChangedFile.csv")
os.remove("TestFile.csv")
os.remove("TestCSV.csv")
os.remove("FormatLib.py")

print("Files Removed!")
