print ("ilk ödev")
v_message = "hello friends"



print("hi")
print(v_message)
v_name1 = "zeynep"

v_name2 = "sude"

v_surname = "dinler"



v_fullname = v_name1 + v_name2 + v_surname

print(v_fullname)
v_fullname = v_name1 + " " + v_name2 + " " + v_surname

print(v_fullname)
v_num1 = "500"

v_num2 = "300"

v_numSum1 = v_num1 + v_num2

print(v_numSum1)
#length



v_lenFull = len(v_fullname)

print("v_fullname:" ,v_fullname, "and lengt is: " ,v_lenFull)
v_titleF = v_fullname.title()

print("v_fullname :" ,v_fullname, "and title is" , v_titleF)
#upper:

v_upperF = v_fullname.upper()



#lower

v_lowerF = v_fullname.lower()

print("v_fullname:" , v_fullname ,"Upper:" ,v_upperF , " Lower " ,v_lowerF)

v_ch2 = v_fullname [7]

print(v_ch2)
v_num1 = 500

v_num2 = 300

v_sum1 = v_num1 + v_num2



print(v_sum1, " and type :" , type(v_sum1))
v_num1 = v_num1 + 72

v_num2 = v_num2 - 25.5

v_sum1 = v_num1 + v_num2 



print(v_sum1)
print("v_sum1 : " ,v_sum1 , "type :" ,type(v_sum1))
v_fl1 = 56.5

v_fl2 = 18.5

v_s3 = v_fl1 + v_fl2



print(v_s3 , type(v_s3))
def z_SayGreen():

    print("Hi. I am coming from z_SayGreen")

    

def z_SayGreen2():

    print("Hi. I am coming from z_SayGreen2")

    print("Good")

    

z_SayGreen()
z_SayGreen2()
def z_BlueMessages(z_BlueMessages):

    print(z_BlueMessages , "came from 'z_BlueMessages'")

    

def z_getFullName(z_FirstName , z_Surname , z_Age):

    print("Welcome" , z_FirstName , " " , z_Surname , "your age :" , z_Age)
z_BlueMessages("selam selam selam")
z_getFullName("Zeynep" , "DİNLER" , 15)
def z_Calc1(z_Num1 , z_Num2 , z_Num3 , z_Num4):

    z_Answer = z_Num1 + z_Num2 + z_Num3 + z_Num4

    print("Answer = " , " " , z_Answer)
z_Calc1(60 , 60 , 40 , 20)
# return function

def z_Calc2(z_Num1 , z_Num2 , z_Num3 , z_Num4):

    z_Out = z_Num1 + z_Num2 + z_Num3 * z_Num4

    print ("Hi from f_Calc2")

    return z_Out
z_incoming = z_Calc2(1,2,3,4)

print("Score is : " , z_incoming)
# Default Functions :



def z_getSchoolInfo(z_Name , z_StudentCount , z_City = "ISTANBUL"):

    print("Name : " , z_Name , "St Count : " , z_StudentCount , "City : " , z_City)

    
z_getSchoolInfo("AAIHL" , 482)

z_getSchoolInfo("KAIHL" , 578 , "KAGITHANE")
# Flexible Functions :



def z_Flex1(z_Name , *z_messages):

    print("Hi " , z_Name , "Your first messages is" , z_messages [1])
z_Flex1("Milan", "Amoureux" , "Selam")
# Lambda Function :



z_result1 = lambda x : x*4

print("Result is : " , z_result1(6))
def z_alan(kenar1,kenar2):

    print(kenar1*kenar2)
z_alan(3,5)
s_list1 = [1,2,3,4,5,6,8]

print(s_list1)

print("Type for 's_list1' is : " , type (s_list1))
z_list1_4 = s_list1[4]

print(z_list1_4)

print("Type of 'z_list1_4' is : " , type(z_list1_4))
s_list2 = ["Harry" , "Ron" , "Hermonie" , "Snape" , "Draco" , "Hagrid"]

print(s_list2)

print("Type of 's_list2' is : " , type(s_list1))
z_list2_4 = s_list2[2]

print(z_list2_4)

print("Type of 'v_list2_4' is : " , type(z_list2_4))
z_list2_x4 = s_list2[-4]

print(z_list2_x4)
s_list2_2 = s_list2[0:6]

print(s_list2_2)
#Len

z_len_s_list2_2 = len(s_list2_2)

print("Size of 's_list2_2' is : " , z_len_s_list2_2)

print(s_list2_2)
#Append

s_list2_2.append("Ron")

print(s_list2_2)



s_list2_2.append("Harry")

print(s_list2_2)
#Reverse

s_list2_2.reverse()

print(s_list2_2)
s_list2_2.sort()

print(s_list2_2)
#Remove



#First add 'Hagrid' then Remove 'Hagrid'

s_list2_2.append('Hagrid')

print(s_list2_2)
s_list2_2.remove("Hagrid")

print(s_list2_2)
z_dict1 = {"blue": "mavi" , "green": "yeşil" , "black": "siyah"}



print(z_dict1)

print(type(z_dict1))
z_green = z_dict1["green"]



print(z_green)

print(type(z_green))
#Keys & Values



z_keys = z_dict1.keys()

z_values = z_dict1.values()



print(z_keys)

print(type(z_keys))



print()

print(z_values)

print(type(z_values))
z_reux1 = 20

z_reux2 = 30



if z_reux1 > z_reux2:

    print(z_reux1 , " is greater then " , z_reux2)

elif z_reux1 < z_reux2:

    print(z_reux1 , " is smaller then " , z_reux2)

else:

    print("This 2 variables are equal")
# < , <= , > , >= , == , <>

def z_Comparison1(z_nina1 , z_nina2):

    if z_nina1 >z_nina2:

        print(z_nina2 , " is greater then " , z_nina2)

    elif z_nina1 < z_nina2:

        print(z_nina1 , " is smaller then " , z_nina2)

    else :

        print("These " , z_nina2 , " variables are equal")

        

z_Comparison1(55,88)

z_Comparison1(77,33)

z_Comparison1(24,24)
def f_IncludeOrNot(z_search, z_searchList):

    if z_search in z_searchList :

        print("Good news ! ",z_search , " is in list.")

    else :

        print(z_search , " can't reach, help me! ")



s_list = list(z_dict1.keys())

print(s_list)

print(type(s_list))



f_IncludeOrNot("green" , s_list)

f_IncludeOrNot("pink" , s_list)
for z in range (0,16):

    print("o", z , "yaşında.")
z_tiredMessage = "ALWAYS TIRED"

print(z_tiredMessage)
for z_chrs in z_tiredMessage:

    print(z_chrs)

    print(" '''''''' ")
for z_chrs in z_tiredMessage.split():

    print(z_chrs)
print(s_list1)

s_sum_list1 = sum(s_list1)

print("Sum of l_list1 is : " , s_sum_list1)



print()

z_cum_list1 = 0

z_loopindex = 0

for z_current in s_list1:

    z_cum_list1 = z_cum_list1 + z_current

    print(z_loopindex , " nd value is : " , z_current)

    print("Cumulative is : " , z_cum_list1)

    z_loopindex = z_loopindex + 1

    print("------")

z = 0

while(z < 7):

    print("o" , z , "yaşında.")

    z = z+1
print(s_list1)

print()



c = 0

k = len(s_list1)



while(c<k):

    print(s_list1[c])

    c=c+1
#Let's find minimum and maximum number in list



s_list2 = [2,6,8,-7,-200,355,81,43,-95]



z_min = 0

z_max = 0



z_index = 0

z_len = len(s_list2)



while (z_index < z_len):

    z_current = s_list2[z_index]

    

    if z_current > z_max:

        z_max = z_current

    

    if z_current < z_min:

        z_min = z_current

    

    z_index = z_index+1



print ("Maximum number is : " , z_max)

print ("Minimum number is : " , z_min)
# Import library to use

import numpy as np
z_array1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

z_array2_np = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print("z_array1 : " , z_array1)

print("Type of z_array1 : " , type(z_array1))
print("z_array2_np : " , z_array2_np)

print("Type of z_array2_np : " , type(z_array2_np))
# shape

z_shape1 = z_array2_np.shape

print("z_shape1 : " , z_shape1 , " and type is : " , type(z_shape1))
# Reshape

z_array3_np = z_array2_np.reshape(3,5)

print(z_array3_np)
z_shape2 = z_array3_np.shape

print("z_shape2 : " , z_shape2 , " and type is : " , type(z_shape2))
# Dimension

z_reux1 = z_array3_np.ndim

print("z_reux1 : " , z_reux1 , " type is : " , type(z_reux1))
# Dtype.name

z_archerd1 = z_array3_np.dtype.name

print("z_archerd1 : " , z_archerd1 , " and type is : " , type(z_archerd1))
# Size

z_size1 = z_array3_np.size

print("z_size1 : " , z_size1 , " and type : " , type(z_size1))

# Let's create 3x4 array

z_array4_np = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

print(z_array4_np)

print("---------------")

print("Shape is : " , z_array4_np.shape)
# Zeros

z_array5_np = np.zeros((5,7))

print(z_array5_np)
# Update an item on this array 

z_array5_np[0,5] = 45

print(z_array5_np)
# Ones

z_array6_np = np.ones((6,9))

print(z_array6_np)
# Empty

z_array7_np = np.empty((8,5))

print(z_array7_np)
# Arrange

z_array8_np = np.arange(20,60,6)

print(z_array8_np)

print(z_array8_np.shape)
# Linspace

z_array9_np = np.linspace(5,60,5)

z_array10_np = np.linspace(30,20,10)



print(z_array9_np)

print(z_array9_np.shape)

print("-----------------------")

print(z_array10_np)

print(z_array10_np.shape)
# Sum , Subtract , Square

z_np1 = np.array([4,5,6])

z_np2 = np.array([7,8,9])



print(z_np1 + z_np2)

print(z_np1 - z_np2)

print(z_np2 - z_np1)

print(z_np1 ** 2)
# Sinus

print(np.sin(z_np2))
# True / False

z_np2_TF = z_np2 < 8

print(z_np2_TF)

print(z_np2_TF.dtype.name)
# Element wise Prodcut

z_np1 = np.array([1,2,3])

z_np2 = np.array([7,8,9])

print(z_np1 * z_np2)
# Transpose

z_np5 = np.array([[2,4,8],[3,6,1]])

z_np5Transpose = z_np5.T

print(z_np5)

print(z_np5.shape)

print()

print(z_np5Transpose)

print(z_np5Transpose.shape)
 #Matrix Multiplication

z_np6 = z_np5.dot(z_np5Transpose)

print(z_np6)
# Exponential --> We will use on Statistics Lesson

z_np5Exp = np.exp(z_np5)



print(z_np5)

print(z_np5Exp)
# Random 

z_np8 = np.random.random((7,5)) # --> It will get between 0 and 1 random numbers

print(z_np8)
#Sum , Max ,Min

z_np8Sum = z_np8.sum()

print("Sum of array : ", z_np8Sum)  #--> Remember ! If you get sum of array we can use that :  sum(array1)

print("Max of array : ", z_np8.max()) #--> Remember ! If you get max of array we can use that :  max(array1)

print("Min of array : ", z_np8.min()) #--> Remember ! If you get min of array we can use that :  min(array1)
# Sum with Column or Row

print("Sum of Columns :")

print(z_np8.sum(axis=0)) # --> Sum of Columns

print()

print("Sum of Rows :")

print(z_np8.sum(axis=1)) #Sum of Rows
# Square , Sqrt

print(np.sqrt(z_np8))

print()

print(np.square(z_np8))
# Add

z_np10 = np.array([1,2,3,4,5])

z_np11 = np.array([10,20,30,40,50])



print(np.add(z_np10,z_np11))
z_np12 = np.array([1,2,3,4,5,6,7,8,9])



print("First item is : " , z_np12[0])

print("Third item is : " , z_np12[2])
# Get top 4 rows :

print(z_np12[0:4])
# Reverse

z_np12_Rev = z_np12[::-1]

print(z_np12_Rev)
z_np13 = np.array([[1,2,3,4,5],[11,12,13,14,15]])

print(z_np13)

print()

print(z_np13[1,3]) #--> Get a row



print()

z_np13[1,3] = 314 #--> Update a row

print(z_np13)
# Get all rows but 3rd columns :

print(z_np13[:,2])
#Get 2nd row but 2,3,4th columns

print(z_np13[1,1:4])
# Get last row all columns

print(z_np13[-1,:])
# Get last columns but all rows

print(z_np13[:,-1])
#Flatten

z_np14 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

z_np15 = z_np14.ravel()



print(z_np14)

print("Shape of z_np14 is : " ,z_np14.shape)

print()

print(z_np15)

print("Shape of z_np15 is : " ,z_np15.shape)

print()
# Reshape

z_np16 = z_np15.reshape(3,4)

print(z_np16)

print("Shape of z_np16 is : " ,z_np16.shape)
z_np17 = z_np16.T

print(z_np17)

print("Shape of z_np17 is : " ,z_np17.shape)
z_np20 = np.array([[1,2],[3,4],[5,6]])



print(z_np20)

print()

print(z_np20.reshape(2,3))



print()

print(z_np20) #--> It has not changed !!
# Resize

z_np20.resize((2,3))

print(z_np20) # --> Now it changed !  Resize can change its shape
z_np21 = np.array([[1,2],[3,4]])

z_np22 = np.array([[5,6],[7,8]])



print(z_np21)

print()

print(z_np22)
# Vertical Stack

z_np23 = np.vstack((z_np21,z_np22))

z_np24 = np.vstack((z_np22,z_np21))



print(z_np23)

print()

print(z_np24)

# Horizontal Stack

z_np25 = np.hstack((z_np21,z_np22))

z_np26 = np.hstack((z_np22,z_np21))



print(z_np25)

print()

print(z_np26)
v_list1 = [1,2,3,4]

z_np30 = np.array(v_list1)



print(z_list1)

print("Type of list : " , type(z_list1))

print()

print(z_np30)

print("Type of v_np30 : " , type(z_np30))
v_list2 = list(z_np30)

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