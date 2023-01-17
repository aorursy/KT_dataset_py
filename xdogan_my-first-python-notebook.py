v_message = "Hi. How are you ? "



v_name = "abdüssamed"

v_surname = "doğan"

v_fullname = v_name + " " + v_surname



v_var1 = "10"

v_var2 = "50"

v_varSum = v_var1 + v_var2



print(v_message)



print()

print(v_fullname)



print()

print("v_varSum : " , v_varSum)
#len function :

print("Length of v_fullname is " , len(v_fullname))



#title function :

print()

print("Full Name is " , v_fullname.title())



#upper function :

print()

print("Upper of Full Name is " , v_fullname.upper())



#lower function :

print()

print("Full Name is " , v_fullname.lower())



#type

print()

print("Type of v_fullname is " , type(v_fullname))
v_chr1 = v_fullname[1]

v_chr2 = v_fullname[8]



print("v_chr1 : " , v_chr1 , " and v_chr2 : " , v_chr2)
# Integer

v_num1 = 12.5

v_num2 = 20

v_numSum = v_num1 + v_num2



print("v_num1 : " , v_num1 , " and type : " , type(v_num1))



print()

print("Sum of Num1 and Num2 is : " , v_numSum , " and type : " , type(v_numSum))
# Float

v_num3 = 30.5

v_numSum2 = v_num3 + v_num2



print("v_num3 : " , v_num3 , " and type : " , type(v_num3))



print()

print("Sum of Num2 and Num3 is : " , v_numSum2 , " and type : " , type(v_numSum2))
# Integer

v_num1 = 12

v_num2 = 20

v_numSum = v_num1 + v_num2



print("v_num1 : " , v_num1 , " and type : " , type(v_num1))



print()

print("Sum of Num1 and Num2 is : " , v_numSum , " and type : " , type(v_numSum))
# Float

v_num3 = 30.5

v_numSum2 = v_num3 + v_num2



print("v_num3 : " , v_num3 , " and type : " , type(v_num3))



print()

print("Sum of Num2 and Num3 is : " , v_numSum2 , " and type : " , type(v_numSum2))
def f_SayHello():

    print("Hello")

    

f_SayHello()
def f_SayMessage(v_message):

    print(v_message)

    

f_SayMessage("I am Doğan")

f_SayHello2()
f_getFullName("Abdüssamed" , "DOĞAN" , 15)
def f_Calc1(f_Num1 , f_Num2 , f_Num3):

    a_Sonuc = f_Num1 + f_Num2 + f_Num3

    print("Sonuç =", a_Sonuc)

    
f_Calc1(100 , 250 , 50)
# return function

def f_Calc2(a_Num1 , a_Num2 , a_Num3):

    a_Out = a_Num1+a_Num2+a_Num3*2

    print("Hi from f_Calc2")

    return a_Out

    
a_gelen =  f_Calc2(1,2,3)

print("Score is : " , a_gelen)
# Default Functions :

def f_getSchoolInfo(a_Name,a_StudentCount,a_City = "ISTANBUL"):

    print("Name : " , a_Name , " St Count : " , a_StudentCount 

          , " City : " , a_City)
f_getSchoolInfo("Ayazağa AİHL" , 269)

f_getSchoolInfo("Ankara AİHL" , 432 , "ANKARA")

# Flexible Functions :



def f_Flex1(a_Name , *a_messages):

    

    print ( "Hi" , a_Name , " your first message is :" , a_messages[1] )

f_Flex1("Doğan" , "hi" , "Hello" , "How are you ?")
# Lambda Function :



a_result1 = lambda x : x*3

print("Result is : " , a_result1(3))
def f_alan(kenar1,kenar2):

    print(kenar1*kenar2)
f_alan(7,12)
l_list1 = [9,7,5,3,1,0]

print(l_list1)

print("Type of 'l_list1' is : " , type(l_list1))
v_list1_4 = l_list1[3]

print(v_list1_4)

print("Type of 'v_list1_4' is : " , type(v_list1_4))
l_list2 = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

print(l_list2)

print("Type of 'l_list2' is : " , type(l_list1))
v_list1_4 = l_list1[3]

print(v_list1_4)

print("Type of 'v_list1_4' is : " , type(v_list1_4))

v_list2_x3 = l_list2[-3]

print(v_list2_x3)
l_list2_2 = l_list2[0:4]

print(l_list2_2)
#Append

l_list2_2.append("Saturday")

print(l_list2_2)



l_list2_2.append("Tuesday")

print(l_list2_2)



#Len

v_len_l_list2_2 = len(l_list2_2)

print("Size of 'l_list2_2' is : ",v_len_l_list2_2)

print(l_list2_2)
#Reverse

l_list2_2.reverse()

print(l_list2_2)
#Sort

l_list2_2.sort()

print(l_list2_2)
#Remove



#First add 'Saturday' then Remove 'Tuesday'

l_list2_2.append("Tuesday")

print(l_list2_2)
l_list2_2.remove("Saturday")

print(l_list2_2)
d_dict1 = {"Pencil":"Kalem" , "Short" : "Kısa" , "Lenght": "Uzun", "Teacher": "Öğretmen / Hoca"}



print(d_dict1)

print(type(d_dict1))
v_short = d_dict1["Short"]

print(v_short)

print(type(v_short))
#Keys & Values



v_keys = d_dict1.keys()

v_values = d_dict1.values()





print(v_keys)

print(type(v_keys))



print()

print(v_values)

print(type(v_values))
v_var1 = 12

v_var2 = 24



if v_var1 > v_var2:

    print(v_var1 , " is greater then " , v_var2)

elif v_var1 < v_var2:

    print(v_var1 , " is smaller then " , v_var2)

else :

    print("This 2 variables are equal")
# < , <= , > , >= , == , <>

def f_Comparison1(v_Comp1 , v_Comp2):

    if v_Comp1 > v_Comp2:

        print(v_Comp1 , " is greater then " , v_Comp2)

    elif v_Comp1 < v_Comp2:

        print(v_Comp1 , " is smaller then " , v_Comp2)

    else :

        print("These " , v_Comp1 , " variables are equal")

        

f_Comparison1(22,44)

f_Comparison1(66,33)

f_Comparison1(85,85)
# using 'IN' with LIST





def f_IncludeOrNot(v_search, v_searchList):

    if v_search in v_searchList :

        print("I found what you were looking for! ",v_search , " is in list.")

    else :

        print(v_search , " is not in list. Sorry :(")



l_list = list(d_dict1.keys())

print(l_list)

print(type(l_list))



f_IncludeOrNot("Lenght" , l_list)

f_IncludeOrNot("School" , l_list)
for a in range(0,10):

    print("Hello " , a)
v_loveMessage = "I Love You"

print(v_loveMessage)
for v_chrs in v_loveMessage:

    print(v_chrs)

    print("------")
for v_chrs in v_loveMessage.split():

    print(v_chrs)
print(l_list1)

v_sum_list1 = sum(l_list1)

print("Sum of l_list1 is : " , v_sum_list1)



print()

v_cum_list1 = 0

v_loopindex = 0

for v_current in l_list1:

    v_cum_list1 = v_cum_list1 + v_current

    print(v_loopindex , " nd value is : " , v_current)

    print("Cumulative is : " , v_cum_list1)

    v_loopindex = v_loopindex + 1

    print("------")
i = 0

while(i < 4):

    print("Hi" , i)

    i = i+1
print(l_list1)

print()



i = 0

k = len(l_list1)



while(i<k):

    print(l_list1[i])

    i=i+1
#Let's find minimum and maximum number in list



l_list2 = [3,5,7,-6,-100,255,71,34,-85]



v_min = 0

v_max = 0



v_index = 0

v_len = len(l_list2)



while (v_index < v_len):

    v_current = l_list2[v_index]

    

    if v_current > v_max:

        v_max = v_current

    

    if v_current < v_min:

        v_min = v_current

    

    v_index = v_index+1



print ("Maximum number is : " , v_max)

print ("Minimum number is : " , v_min)
# Import library to use

import numpy as np
v_array1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

v_array2_np = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print("v_array1 : " , v_array1)

print("Type of v_array1 : " , type(v_array1))
print("v_array2_np : " , v_array2_np)

print("Type of v_array2_np : " , type(v_array2_np))
# shape

v_shape1 = v_array2_np.shape

print("v_shape1 : " , v_shape1 , " and type is : " , type(v_shape1))
# Reshape

v_array3_np = v_array2_np.reshape(5,3)

print(v_array3_np)
v_shape2 = v_array3_np.shape

print("v_shape2 : " , v_shape2 , " and type is : " , type(v_shape2))
# Dimension

v_dimen1 = v_array3_np.ndim

print("v_dimen1 : " , v_dimen1 , " type is : " , type(v_dimen1))
# Dtype.name

v_dtype1 = v_array3_np.dtype.name

print("v_dtype1 : " , v_dtype1 , " and type is : " , type(v_dtype1))
# Size

v_size1 = v_array3_np.size

print("v_size1 : " , v_size1 , " and type : " , type(v_size1))
# Let's create 3x4 array

v_array4_np = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

print(v_array4_np)

print("---------------")

print("Shape is : " , v_array4_np.shape)
# Zeros

v_array5_np = np.zeros((4,7))

print(v_array5_np)
# Update an item on this array 

v_array5_np[2,5] = 78

print(v_array5_np)
# Ones

v_array6_np = np.ones((5,3))

print(v_array6_np)
# Empty

v_array7_np = np.empty((2,3))

print(v_array7_np)
# Arrange

v_array8_np = np.arange(52,68,2)

print(v_array8_np)

print(v_array8_np.shape)
# Linspace

v_array9_np = np.linspace(10,30,5)

v_array10_np = np.linspace(10,30,20)



print(v_array9_np)

print(v_array9_np.shape)

print("-----------------------")

print(v_array10_np)

print(v_array10_np.shape)
# Sum , Subtract , Square

v_np1 = np.array([1,9,3])

v_np2 = np.array([5,8,2])



print(v_np1 + v_np2)

print(v_np1 - v_np2)

print(v_np2 - v_np1)

print(v_np1 ** 2)
# Sinus

print(np.sin(v_np2))
# True / False

v_np2_TF = v_np2 < 8

print(v_np2_TF)

print(v_np2_TF.dtype.name)
# Element wise Prodcut

v_np1 = np.array([1,9,3])

v_np2 = np.array([5,8,2])

print(v_np1 * v_np2)
# Transpose

v_np5 = np.array([[2,4,8],[3,6,1]])

v_np5Transpose = v_np5.T

print(v_np5)

print(v_np5.shape)

print()

print(v_np5Transpose)

print(v_np5Transpose.shape)
# Matrix Multiplication

v_np6 = v_np5.dot(v_np5Transpose)

print(v_np6)
# Exponential --> We will use on Statistics Lesson

v_np5Exp = np.exp(v_np5)



print(v_np5)

print(v_np5Exp)
# Random 

v_np8 = np.random.random((6,6)) # --> It will get between 0 and 1 random numbers

print(v_np8)
#Sum , Max ,Min

v_np8Sum = v_np8.sum()

print("Sum of array : ", v_np8Sum)

print("Max of array : ", v_np8.max())

print("Min of array : ", v_np8.min())
# Sum with Column or Row

print("Sum of Columns :")

print(v_np8.sum(axis=0)) # --> Sum of Columns

print()

print("Sum of Rows :")

print(v_np8.sum(axis=1)) #Sum of Rows
# Square , Sqrt

print(np.sqrt(v_np8))

print()

print(np.square(v_np8))
# Add

v_np10 = np.array([1,4,7,10,13])

v_np11 = np.array([10,20,30,40,50])



print(np.add(v_np10,v_np11))
v_np12 = np.array([1,2,3,4,5,6,7,8,9])



print("First item is : " , v_np12[0])

print("Third item is : " , v_np12[2])
# Get top 4 rows :

print(v_np12[0:4])
# Reverse

v_np12_Rev = v_np12[::-1]

print(v_np12_Rev)
v_np13 = np.array([[1,2,3,4,5],[11,12,13,14,15]])

print(v_np13)

print()

print(v_np13[1,3]) #--> Get a row



print()

v_np13[1,3] = 314 #--> Update a row

print(v_np13)
# Get all rows but 3rd columns :

print(v_np13[:,2])
#Get 2nd row but 2,3,4th columns

print(v_np13[1,1:4])
# Get last row all columns

print(v_np13[-1,:])
# Get last columns but all rows

print(v_np13[:,-1])
#Flatten

v_np14 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

v_np15 = v_np14.ravel()



print(v_np14)

print("Shape of v_np14 is : " ,v_np14.shape)

print()

print(v_np15)

print("Shape of v_np15 is : " ,v_np15.shape)

print()
# Reshape

v_np16 = v_np15.reshape(3,4)

print(v_np16)

print("Shape of v_np16 is : " ,v_np16.shape)
v_np17 = v_np16.T

print(v_np17)

print("Shape of v_np17 is : " ,v_np17.shape)
v_np20 = np.array([[1,2],[3,4],[5,6]])



print(v_np20)

print()

print(v_np20.reshape(2,3))



print()

print(v_np20) #--> It has not changed !!
# Resize

v_np20.resize((2,3))

print(v_np20) # --> Now it changed !  Resize can change its shape
v_np21 = np.array([[1,2],[3,4]])

v_np22 = np.array([[5,6],[7,8]])



print(v_np21)

print()

print(v_np22)
# Vertical Stack

v_np23 = np.vstack((v_np21,v_np22))

v_np24 = np.vstack((v_np22,v_np21))



print(v_np23)

print()

print(v_np24)
# Horizontal Stack

v_np25 = np.hstack((v_np21,v_np22))

v_np26 = np.hstack((v_np22,v_np21))



print(v_np25)

print()

print(v_np26)
v_list1 = [1,2,3,4]

v_np30 = np.array(v_list1)



print(v_list1)

print("Type of list : " , type(v_list1))

print()

print(v_np30)

print("Type of v_np30 : " , type(v_np30))
v_list2 = list(v_np30)

print(v_list2)

print("Type of list2 : " , type(v_list2))
v_list3 = v_list2

v_list4 = v_list2



print(v_list2)

print()

print(v_list3)

print()

print(v_list4)
v_list2[0] = 55



print(v_list2)

print()

print(v_list3) # --> Same address with list2

print()

print(v_list4) # --> Same address with list2
v_list5 = v_list2.copy()

v_list6 = v_list2.copy()



print(v_list5)

print()

print(v_list6)
v_list2[0] = 71



print(v_list2)

print()

print(v_list5) # --> Not same address with list2

print()

print(v_list6) # --> Not same address with list2
# Import Library

import pandas as pd
# Let's create Data Frame from Dictionary

v_dict1 = { "COUNTRY" : ["TURKEY","AFGHANİSTAN","GERMANY","FRANCE","U.S.A","AZERBAIJAN","IRAN"],

            "CAPITAL":["ISTANBUL","KABUL","BERLIN","PARIS","NEW YORK","BAKU","TAHRAN"],

            "POPULATION":[15.07,5.11,3.57,2.12,8.62,4.3,8.69]}



v_dataFrame1 = pd.DataFrame(v_dict1)



print(v_dataFrame1)

print()

print("Type of v_dataFrame1 is : " , type(v_dataFrame1))
# get top 5 rows

v_head1 = v_dataFrame1.head()

print(v_head1)

print()

print("Type of v_head1 is :" ,type(v_head1))
# get top 100 rows

print(v_dataFrame1.head(100))
# get last 5 rows

v_tail1 = v_dataFrame1.tail()

print(v_tail1)

print()

print("Type of v_tail1 is :" ,type(v_tail1))
# Columns

v_columns1 = v_dataFrame1.columns

print(v_columns1)

print()

print("Type of v_columns is : " , type(v_columns1))
v_info1 = v_dataFrame1.info()

print(v_info1)

print()

print("Type of v_info1 is : " , type(v_info1))
v_dtypes1 = v_dataFrame1.dtypes

print(v_dtypes1)

print()

print("Type of v_dtypes1 is : " , type(v_dtypes1))
v_descr1 = v_dataFrame1.describe()

print(v_descr1)

print()

print("Type of v_descr1 is : " , type(v_descr1))
v_country1 = v_dataFrame1["POPULATION"]

print(v_country1)

print()

print("Type of v_country1 is : " , type(v_country1))
# Add new column

v_currenyList1 = ["TRY","GBP","EUR","EUR","USD","AZN","IRR"]

v_dataFrame1["CURRENCY"] = v_currenyList1



print(v_dataFrame1.head())
# Get all rows ,  1 column

v_AllCapital = v_dataFrame1.loc[:,"CAPITAL"]

print(v_AllCapital)

print()

print("Type of v_AllCapital is : " , type(v_AllCapital))
# Get top 3 rows of Currency

v_top3Currency = v_dataFrame1.loc[0:3,"CURRENCY"]

print(v_top3Currency)
v_CityCountry = v_dataFrame1.loc[:,["CAPITAL","COUNTRY","BLABLA"]] #--> BLABLA not defined !!!

print(v_CityCountry)
v_Reverse1 = v_dataFrame1.loc[::-1,:]

print(v_Reverse1)
print(v_dataFrame1.loc[:,:"POPULATION"])

print()

print(v_dataFrame1.loc[:,"POPULATION":])
#Get data with column index (not column name)

print(v_dataFrame1.iloc[:,2])
v_filter1 = v_dataFrame1.POPULATION > 4

print(v_filter1)
v_filter2 = v_dataFrame1["POPULATION"] < 9

print(v_filter2)
print(v_dataFrame1[v_filter1 & v_filter2])
print(v_dataFrame1[v_dataFrame1["CURRENCY"] == "TRY"])

v_meanPop =v_dataFrame1["POPULATION"].mean()

print(v_meanPop)



v_meanPopNP = np.mean(v_dataFrame1["POPULATION"])

print(v_meanPopNP)
for a in v_dataFrame1["CURRENCY"]:

    print(a)
v_dataFrame1["POP LEVEL"] = ["Low" if v_meanPop > a else "HIGH" for a in v_dataFrame1["POPULATION"]]

print(v_dataFrame1)
print(v_dataFrame1.columns)



v_dataFrame1.columns = [a.lower() for a in v_dataFrame1.columns]



print(v_dataFrame1.columns)
v_dataFrame1.columns = [a.split()[0]+"_"+a.split()[1] if (len(a.split())>1) else a for a in v_dataFrame1.columns]

print(v_dataFrame1.columns)
v_dataFrame1["test1"] = [5,10,15,20,25,30,35]

print(v_dataFrame1)
print(v_dataFrame1)
v_data1 = v_dataFrame1.head()

v_data2 = v_dataFrame1.tail()



print(v_data1)

print()

print(v_data2)
v_dataConcat1 = pd.concat([v_data1,v_data2],axis=0) # axis = 0 --> VERTICAL CONCAT

v_dataConcat2 = pd.concat([v_data2,v_data1],axis=0) # axis = 0 --> VERTICAL CONCAT



print(v_dataConcat1)

print()

print(v_dataConcat2)
v_CAPITAL = v_dataFrame1["capital"]

v_POPULATION = v_dataFrame1["population"]



v_dataConcat3 = pd.concat([v_CAPITAL,v_POPULATION],axis=1) #axis = 1 --> HORIZONTAL CONCAT

v_dataConcat4 = pd.concat([v_POPULATION,v_CAPITAL],axis=1) #axis = 1 --> HORIZONTAL CONCAT

print(v_dataConcat3)

print()

print(v_dataConcat4)
v_dataFrame1["population"] = [a*2 for a in v_dataFrame1["test1"]]

print(v_dataFrame1)
def f_multiply(v_population):

    return v_population*3



v_dataFrame1["test2"] = v_dataFrame1["population"].apply(f_multiply)

print(v_dataFrame1)