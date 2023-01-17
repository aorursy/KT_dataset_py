print("Hello World")

v_message = "Hello World"



print("Hi")
print(v_message)
v_name = "yasin"

v_surname = "tavman"



v_fullname = v_name + v_surname

print(v_fullname)
v_fullname = v_name + " " + v_surname



print(v_fullname)
v_num1 = "500"

v_num2 = "200"

v_numSum1 = v_num1 + v_num2

print(v_numSum1)
#lenght  

v_lenFull = len(v_fullname)

print("v_fullname : ", v_fullname,"and lenght is : " , v_lenFull)
#upper

v_upperF = v_fullname.upper()



#lower

v_lowerF = v_fullname.lower()

print("v_fullname : " , v_fullname , "Upper : " ,v_upperF , "Lower : " , v_lowerF)
v_2ch = v_fullname[1]

print(v_2ch)
v_num1=200

v_num2=750

v_sum1=v_num1 + v_num2



print(v_sum1 , " and type : " , type(v_sum1))
v_num1 = v_num1 + 90

v_num2 = v_num2 + 15.2

v_sum1 = v_num1 + v_num2



print(v_num1)
print("v_sum1 : ",v_sum1 , " type : ", type(v_sum1))
v_fl1 = 35.4

v_fl2 = 45.3

v_s3 = v_fl1 + v_fl2



print(v_s3 , type(v_s3))
def f_SayHello():

    print("Hi. I am coming from f_SayHello")

    

def f_SayHello2():

    print("hi.I am coming from f_SayHello2")

    print("Good")

    

f_SayHello()    
f_SayHello2()
def f_sayMessage(v_Message1):

    print(v_Message1 , "came from 'f_sayMessage'")

    

def f_getFullName(v_FirstName , v_Surname , v_Age):

    print("Welcome" , v_FirstName , "" , v_Surname , " your age :" , v_Age)
f_sayMessage("How are you ?")
f_getfullName("Yasin" , "TAVMAN" , 14)
def f_Calc1(f_Num1 , f_Num2 , f_Num3):

    v_Sonuc= f_Num1 + f_Num2 + f_Num3

    print("Sonuç = " ," " , v_Sonuc)
f_Calc1(1000 ,10 ,350)
# return function

def f_Calc2(v_Num1 , v_Num2 , v_Num3):

    v_Out = v_Num1+v_Num2+v_Num3*4

    print("Hi from f_Calv2")

    return v_Out
v_gelen = f_Calc2(1,2,3)

print("Score is :" , v_gelen)
# Default Functions

def f_getSchoolInfo(v_Name,v_StudentCount,v_City = "ISTANBUL"):

    print("Name : " , v_Name , " St Count : " , v_StudentCount , "City :" , v_City)
f_getSchoolInfo("AAIHL" , 521)

f_getSchoolInfo("Ankara Fen" , 521 ,"ANKARA")
#Flexible Functions :



def f_Flex1(v_Name , *v_messages):

    print("Hi " , v_Name , " your first message is :" , v_messages[2])
# Lambda Function



v_result1 = lambda x : x*5

print("result is :" , v_result1(8))
def f_alan(kenar1,kenar2):

    print(kenar1*kenar2)
f_alan(17,15)
l_list1 = [1,5,3,6,8,9]

print(l_list1)

print("Type of 'l_list1' is : " , type(l_list1))
v_list1_4 = l_list1[3]

print(v_list1_4)

print("Type of 'v_list1_4' is : " , type(v_list1_4))
l_list2 = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

print(l_list2)

print("Type of 'l_list2' is : " , type(l_list1))
v_list2_4 = l_list2[3]

print(v_list2_4)

print("Type of 'v_list2_4' is : " , type(v_list2_4))
v_list2_x3 = l_list2[-3]

print(v_list2_x3)
l_list2_2 = l_list2[0:3]

print(l_list2_2)
v_len_l_list2_2 = len(l_list2_2)

print("Size of 'l_list2_2' is : ",v_len_l_list2_2)

print(l_list2_2)
l_list2_2.append("Wednesday")

print(l_list2_2)



l_list2_2.append("Monday")

print(l_list2_2)
#Reverse

l_list2_2.reverse()

print(l_list2_2)
#Sort

l_list2_2.sort()

print(l_list2_2)
#Remove



#First add 'Saturday' then Remove 'Saturday'

l_list2_2.append("Wednesday")

print(l_list2_2)
l_list2_2.remove("Wednesday")

print(l_list2_2)
d_dict1 = {"Home":"Ev" , "School" : "Okul" , "Student": "Öğrenci"}



print(d_dict1)

print(type(d_dict1))
v_school = d_dict1["School"]

print(v_school)

print(type(v_school))








v_keys = d_dict1.keys()

v_values = d_dict1.values()





print(v_keys)

print(type(v_keys))



print()

print(v_values)

print(type(v_values))
v_var1 = 45

v_var2 = 65



if v_var1 > v_var2:

    print(v_var1 , " is greater then " , v_var2)

elif v_var1 < v_var2:

    print(v_var1 , " is smaller then " , v_var2)

else :

    print("This 2 variables are equal")
def f_Comparison1(v_Comp1 , v_Comp2):

    if v_Comp1 > v_Comp2:

        print(v_Comp1 , " is greater then " , v_Comp2)

    elif v_Comp1 < v_Comp2:

        print(v_Comp1 , " is smaller then " , v_Comp2)

    else :

        print("These " , v_Comp1 , " variables are equal")

        

f_Comparison1(77,88)

f_Comparison1(83,83)

f_Comparison1(99,11)








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
for a in range(0,45):

    print("naber " , a)
v_happyMessage = "ben mutluyum"

print(v_happyMessage)
for v_chrs in v_happyMessage:

    print(v_chrs)

    print("------")
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
