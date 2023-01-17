# check the hash method

help(hash)
hash([1,2,3])   #can't use the hash method on a list
hash((1,2,3))   #a tuple can
def problem1(l):

    t = tuple(l)   #convert the list to tuple

    hash_result = hash(t)   #use the builtin hash method

    return hash_result

problem1([1,2,3])
problem1([1,2,3,4])
def problem2(grades,name):

    name_scores = {}    #empty dict

    for grade in grades:

        split_list = grade.split(",")

        #split_list[0] is the name. use name as key, value is a empty list. 

        name_scores[split_list[0]] = []   

        #convert the string to int and append to the list

        name_scores[split_list[0]].append(int(split_list[1]))

        name_scores[split_list[0]].append(int(split_list[2]))

        name_scores[split_list[0]].append(int(split_list[3]))

        #the above three lines could be replaced by the following line, if you can understand.

        #name_scores[split_list[0]] = [int(i) for i in split_list[1:]]

        

    #check the final dict, this line is out of the for loop

    print(name_scores)

    #calculate the mean grade of the INPUT name

    mean_of_name = sum(name_scores[name])/len(name_scores[name])

    output = "the average of %s is %.2f" % (name, mean_of_name)   #format the string

    #or use the following format method

    #output = "the average of {} is {:.2f}".format(name, mean_of_name)

    return output

grades = ["Jake, 99, 70, 55","Dennis, 100, 100, 98"]

problem2(grades,"Jake")
problem2(grades,"Dennis")
def problem3(int_list):

    final_list = []    #create a empty list, use the following codes to find all uniq values

    for i in int_list:

        if(i not in final_list):

            final_list.append(i)

            

    print(final_list)  #check the uniq value list

    final_list.sort()  #sort

    print(final_list)

    return final_list[-2]  #this is the second largest value, think about why?
problem3([1,3,1,4,4,2])
problem3([1,2,3])
def problem5(original_set):

    power_set = []    #empty list

    power_set.append([])   #append an empty list

    #loop index from 0 to len-1, like 0,1,2

    for i in range(0,len(original_set)):

#         print(i,original_set[i])

        #loop reset index from i to len-1, like 1,2

        for j in range(i,len(original_set)):

#             print(i,j,original_set[i:j+1])

            #append slice of list[i:j+1] to the power_set

            power_set.append(original_set[i:j+1])

    power_set.sort()

    return power_set
problem5([1])
problem5([1,2])
problem5([1,2,3])
import csv

#import csv module to read csv file

def problem4(tickers):

    return_dict = {}

    file_location = "../input/" #this is my path, not yours

    for ticker in tickers:

        file_name = ticker+".csv"

        #open file

        ticker_csv = open(file_location+file_name)

        #use csv.reader to read from csv file

        ticker_data = csv.reader(ticker_csv)

        #empty list

        ticker_open = []

        for data in ticker_data:

            #use print to check the data, you can uncomment it to check the output

            #print(data)

            #data[2] is the open price value

            ticker_open.append(data[2])

            

        #remove the first line, it's column tile instead of the price value

        ticker_open = ticker_open[1:]

        #convert the string to float num

        ticker_open = [float(i) for i in ticker_open]

        #use print to check again, uncomment it to check the output

        #print(ticker_open)

        #calculate the mean of open price

        ticker_open_mean = sum(ticker_open)/len(ticker_open)

        #store the value in dict

        return_dict[ticker] = ticker_open_mean



    return(return_dict)

problem4(["AAPL"])
problem4(["AAPL","GOOG"])
import os

os.getcwd()
path = os.path.abspath(os.path.dirname(os.getcwd()+"/../"))

print(path)
for root, dirs, files in os.walk(path):

    print(root)

    print(dirs)

    print(files)