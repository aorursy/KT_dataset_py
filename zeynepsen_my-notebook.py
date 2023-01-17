print("Hello World")
v_mesagge = "Hello World"

print("hi")
print(v_mesagge)
v_name = "zeynep"

v_surname = "sen"



v_fullname = v_name + v_surname

print(v_fullname)
v_fullname = (v_name +" "+ v_surname)



print(v_fullname)
v_num1 = "150"

v_num2 = "300"

v_numSun1 = v_num1 + v_num2



print(v_numSun1)
#lenght

v_lenFull = len(v_fullname)



print("v_fullname : " ,v_fullname, "and lengt is:" ,v_lenFull)
v_titleF = v_fullname.title()

print("v_fullname :",  " v_fullname " ,  " and title is : " , v_titleF)
#upper

v_upperF = v_fullname.upper()



#lower

v_lowerF = v_fullname.lower()

print("v_fullname : "  , v_fullname ,  " Upper : " , v_upperF ,  " Lower : " , v_lowerF)
v_2ch = v_fullname[7]    

print(v_2ch)
v_num1 = 150

v_num2 = 300

v_sum1 = v_num1 + v_num2



print(v_sum1, " and type : " , type(v_sum1))
#it will get error

#v_sum2 = v_num1 + v_name

#print(v_sum2)

v_num1 = v_num1 + 45

v_num2 = v_num2 - 25.5

v_sum1 = v_num1 + v_num2



print(v_num1)

print("v_sum1 : ",v_sum1 ,"type : " , type(v_sum1))
v_fl1 = 21.3

v_fl2 = 14.8

v_s3 = v_fl1 + v_fl2



print(v_s3 , type (v_s3))
def f_SayHello1():

    print("Hi. I am comming from f_SayHello")

    

def f_SayHello2():

    print("Hi. I am comming from f_SayHello2")

    print("good")
f_SayHello2()
def f_sayMessage(v_Message1): 

    print(v_Message1 , " came from 'f_sayMessage'") 

def f_getFullName(v_FirstName , v_Surname , v_Age): 

    print("Welcome " , v_FirstName , " " , v_Surname , " your age : " , v_Age) 
f_sayMessage("How are you ?")
f_getFullName("Zeynep" , "Şen" , 14)
def f_Calc1(f_Num1 , f_Num2 , f_Num3):

    v_Sonuc = f_Num1 + f_Num2 + f_Num3 

    print("Sonuç = " ," " , v_Sonuc) 
f_Calc1(150 , 300 , 40)
# return function

def f_Calc2(v_Num1 , v_Num2 , v_Num3):

    v_Out = v_Num1+v_Num2+v_Num3*5

    print("Hi from f_Calc2") 

    return v_Out 
v_gelen = f_Calc2(1,2,3)

print("Score is : " , v_gelen)
# Default Functions :

def f_getSchoolInfo(v_Name,v_StudentCount,v_City = "ISTANBUL"):

    print("Name : " , v_Name , " St Count : " , v_StudentCount , " City : " , v_City)
f_getSchoolInfo("AAIHL" , 521)

f_getSchoolInfo("Ankara Fen" , 521 , "ANKARA")
# Flexible Functions : 

def f_Flex1(v_Name , *v_messages):

    print("Hi " , v_Name , " your first message is : " , v_messages[1]) 
f_Flex1("Zeynep" , "Selam" , "Naber" , "İyisindir İnşallah")
# Lambda Function :

v_result1 = lambda x : x*3 

print("Result is : " , v_result1(5))
def f_alan(kenar1,kenar2): 

    print(kenar1*kenar2) 
f_alan(2,5)
l_list1 = [1,2,3,4,5,6]

print(l_list1)

print("Type of 'l_list1' is : " , type(l_list1))
v_list1_4 = l_list1[5]

print(v_list1_4)

print("Type of 'v_list1_4' is : " , type(v_list1_4))
l_list2 = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

print(l_list2)

print("Type of 'l_list2' is : " , type(l_list1))
v_list2_4 = l_list2[4]

print(v_list2_4)

print("Type of 'v_list2_4' is : " , type(v_list2_4))
v_list2_x3 = l_list2[-2]

print(v_list2_x3)
l_list2_2 = l_list2[0:3]

print(l_list2_2)
#Len

v_len_l_list2_2 = len(l_list2_2)

print("Size of 'l_list2_2' is : ",v_len_l_list2_2)

print(l_list2_2)
#Append

l_list2_2.append("Saturday")

print(l_list2_2)



l_list2_2.append("Tuesday")

print(l_list2_2)
#Reverse

l_list2_2.reverse()

print(l_list2_2)
#Sort

l_list2_2.sort()

print(l_list2_2)
#Remove



#First add 'Saturday' then Remove 'Saturday'

l_list2_2.append("Saturday")

print(l_list2_2)
l_list2_2.remove("Saturday")

print(l_list2_2)
d_dict1 = {"Home":"Ev" , "School" : "Okul" , "Student": "Öğrenci"}



print(d_dict1)

print(type(d_dict1))
v_student = d_dict1["Student"]

print(v_student)

print(type(v_student))
#Keys & Values



v_keys = d_dict1.keys()

v_values = d_dict1.values()





print(v_keys)

print(type(v_keys))



print()

print(v_values)

print(type(v_values))
v_var1 = 5

v_var2 = 10



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

        

f_Comparison1(15,100)

f_Comparison1(150,120)

f_Comparison1(65,65)
# using 'IN' with LIST





def f_IncludeOrNot(v_search, v_searchList):

    if v_search in v_searchList :

        print("Good news ! ",v_search , " is in list.")

    else :

        print(v_search , " is not in list. Sorry :(")



l_list = list(d_dict1.keys())

print(l_list)

print(type(l_list))



f_IncludeOrNot("Home" , l_list)

f_IncludeOrNot("Pencil" , l_list)
for a in range(1,10):

    print("saat :", a)

v_happyMessage =  "ALWAYS SMILE"

print(v_happyMessage)
for v_chrs in v_happyMessage:

    print(v_chrs)

    print("----------")
for v_chrs in v_happyMessage.split():

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
i = 1

while(i < 5):

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



l_list2 = [2,4,6,-8,-102,200,65,21,-96]



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

v_array5_np = np.zeros((3,4))

print(v_array5_np)
# Update an item on this array 

v_array5_np[0,3] = 30

print(v_array5_np)
# Ones

v_array6_np = np.ones((3,4))

print(v_array6_np)
# Empty

v_array7_np = np.empty((2,3))

print(v_array7_np)
# Arrange

v_array8_np = np.arange(10,45,5)

print(v_array8_np)

print(v_array8_np.shape)
# Linspace

v_array9_np = np.linspace(15,35,10)

v_array10_np = np.linspace(15,35,25)



print(v_array9_np)

print(v_array9_np.shape)

print("-----------------------")

print(v_array10_np)

print(v_array10_np.shape)
# Sum , Subtract , Square

v_np1 = np.array([1,2,3])

v_np2 = np.array([7,8,9])



print(v_np1 + v_np2)

print(v_np1 - v_np2)

print(v_np2 - v_np1)

print(v_np1 ** 2)
# Sinus

print(np.sin(v_np2))
# True / False

v_np2_TF = v_np2 < 20

print(v_np2_TF)

print(v_np2_TF.dtype.name)
# Element wise Prodcut

v_np1 = np.array([1,2,3])

v_np2 = np.array([8,9,10])

print(v_np1 * v_np2)
v_np3 = np.array([5,4,2,1,6,7,8,3,9])



print("First item is : " , v_np3[5])

print("Third item is : " , v_np3[1])

# Get top 4 rows :

print(v_np3[2:8])
# Reverse

v_np3_rev = v_np3[::-1]

print(v_np3_rev)
v_np4 = np.array([[2,4,6,8,10],[1,3,5,7,9]])

print(v_np4)

print()

print(v_np4[0,3]) #--> Get a row



print()

v_np4[1,3] = 566 #--> Ubtade a row

print(v_np4)

# Get all rows but 3rd columns :

print(v_np4[:,4])
#Get 2nd row but 2,3,4th columns

print(v_np4[1,1:3])
# Get last row all columns

print(v_np4[-1,:])
# Get last columns but all rows

print(v_np4[:,-1])
#Flatten

v_np5 = np.array([[5,4,2],[8,7,6],[3,1,9],[12,11,10]])

v_np6 = v_np3.ravel()



print(v_np5)

print("Shape of v_np3 is : " ,v_np5.shape)

print()

print(v_np6)

print("Shape of v_np4 is : " ,v_np6.shape)

print()
# Reshape

v_np7 = v_np6.reshape(3,3)

print(v_np7)

print("Shape of v_np16 is : " ,v_np7.shape)
v_np8 = v_np7.T

print(v_np8)

print("Shape of v_np8 is : " ,v_np8.shape)
v_np9 = np.array([[9,8,7],[6,5,4],[3,2,1]])



print(v_np9)

print()

print(v_np9.reshape(3,3))



print()

print(v_np9) #--> It has not changed !!
# Resize

v_np9.resize((3,3))

print(v_np9) # --> Now it changed !  Resize can change its shape
v_np10 = np.array([[2,4],[3,6]])

v_np11 = np.array([[9,5],[1,7]])



print(v_np10)

print()

print(v_np11)
# Vertical Stack

v_np12 = np.vstack((v_np10,v_np11))

v_np13 = np.vstack((v_np11,v_np10))



print(v_np12)

print()

print(v_np13)
# Horizontal Stack

v_np14 = np.hstack((v_np10,v_np11))

v_np15 = np.hstack((v_np11,v_np10))



print(v_np14)

print()

print(v_np15)
v_list1 = [5,6,7,8]

v_np16 = np.array(v_list1)



print(v_list1)

print("Type of list : " , type(v_list1))

print()

print(v_np16)

print("Type of v_np16 : " , type(v_np16))
v_list2 = list(v_np16)

print(v_list2)

print("Type of list2 : " , type(v_list2))
v_list3 = v_list2

v_list4 = v_list2



print(v_list2)

print()

print(v_list3)

print()

print(v_list4)
v_list2[3] = 86



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
v_list2[1] = 62



print(v_list2)

print()

print(v_list5) # --> Not same address with list2

print()

print(v_list6) # --> Not same address with list2