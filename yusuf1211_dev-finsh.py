print("hello my name is yusuf")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
v_message  = "selamın aleyküm"





print("hi")
print(v_message)
v_name= "yusuf kılıç "

v_fullname= v_name 



print(v_fullname)
v_num1="534"

v_num2="50"

v_numSum1=v_num1 + v_num2

print(v_numSum1)
#lenght





v_lenfull = len(v_fullname)



print("v_fullname:",v_fullname,"and lengt is:"

,v_lenfull)
#upper

v_upperF = v_fullname.upper()



#lower

v_lowerF = v_fullname.lower()

print("v_fullname : " , v_upperF , 

"lower : " , v_lowerF)
def f_SayHello():

    print("hi. I am comming from f_SayHello")

    

def f_SayHello2():

    print("hi. I am coming from f_SayHello2")

print("good")



f_SayHello()
f_SayHello2()
def f_sayMessage(a_Message1):

    print(a_Message1,"came from'f_sayMessge'")

    

def f_getFullName(a_FirstName,a_surname,a_Age):

    print("welcome",a_FirstName," ",a_Surname,"your age:",age)
f_sayMessage("how are you?")
f_getFullName

("yusuf" , "kılıç",14)
def f_Calc1(f_Num1 , f_Num2 , f_Num3):

    a_Sonuc = f_Num1 + f_Num2 + f_Num3

    print("Sonuç =", a_Sonuc)
f_Calc1(100 , 1200 , 590)
# return function

def f_Calc2(a_Num1 , a_Num2 , a_Num3):

    a_Out = a_Num1+a_Num2+a_Num3*2

    print("Hi from f_Calc2")

    return a_Out
e_city=["istanbul","erzurum","denizli","muğla"]



print(e_city)



print("type of city list:",type(e_city))
# Flexible Functions :



def f_Flex1(a_Name , *a_messages):

    

    print ( "Hi" , a_Name , " your first message is :" , a_messages[1] )
f_Flex1("yusuf", "hi" , "Hello" , "How are you ?")
#Lambda Function:



e_example=lambda x: x*31



print("example:",e_example(124))
l_list1 = [1,2,3,4,6,10]

print(l_list1)

print("type of 'l_list1 'is:", type (l_list1))
l_list2 = ['monday', 'tuesday','june', 'Thursday', 'friday','saturday','sunday']



print(l_list2)

print("type of ' l_list2'is:"

,type(l_list2))
r_list2_4= l_list2[2]

print(r_list2_4)

print("type of 'r_list2_4'is:",  

      type(r_list2_4))
r_list2_x3 = l_list2[-6]

print(r_list2_x3)
l_list2_2=l_list2[0:6]

print(l_list2_2)
#len





r_len_l_list2_2=len(l_list2_2)

print("Size of 'l_list2_2'is:",r_len_l_list2_2)

print(l_list2_2)
#append 

l_list2_2.append("JUNE")

print(l_list2_2)



l_list2_2. append("MARCH")

print(l_list2_2)
#Remove



#First add 'Saturday' then Remove 'Saturday'

l_list2_2.append("JUNE")

print(l_list2_2)
l_list2_2.remove("monday")

print(l_list2_2)
e_dict = {"kitap": "defter,masa,araba,ev,pasta"} 
print(e_dict)

print(type(e_dict))
e_deutsch=e_dict["kitap"]

print(e_deutsch)

print(type(e_deutsch))
#keys & values 



e_key=e_dict.keys()

print(e_key)

print(type(e_key))
e_values=e_dict.values()



print(e_values)

print(type(e_values))
v_num1 = 611

v_num2 = 2005

v_num3 = 12



if v_num1 > v_num2 <v_num3:

    print(v_num1 ," is greater then " , v_num2 , " is smaller then " , v_num3)

elif v_num1 < v_num2:

    print(v_num1 , " is smaller then " , v_num2 , "is greater then" , v_num3 )

else :

    print("This 2 variables are equal")
# < , <= , > , >= , == , <>



def e_factor(e_öge1 , e_öge2):

    if e_öge1 > e_öge2:

        print(e_öge1 , " is greater then " , e_öge2)

    elif e_öge1 < e_öge2:

        print(e_öge1 , " is smaller then " , e_öge2)

    else :

        print("These " , e_öge1 , " variables are equal")

        

e_factor(55,86)

e_factor(50,25)

e_factor(90,79)
for n_numbers in range(1,14):

    print(n_numbers,'yasındaydım')
v_m = '14 yaşıma girdim.'

print(v_m)

for v_c in v_m:

    print(v_c)

    print("******")
l_list1=[1,10,100,31,76]

print(l_list1)

v_sum_list1 = sum(l_list1)

print("listenin toplamı : " , v_sum_list1)



print()

v_cum_list1 = 0

v_loopindex = 0

for v_current in l_list1:

    v_cum_list1 = v_cum_list1 + v_current

    print(v_loopindex , " . degisken : " , v_current)

    print("Toplamı : " , v_cum_list1)

    v_loopindex = v_loopindex + 1

    print("------")

[1,10 ,100 ,31 ,76 ]
print(l_list1)

print()



i = 0

k = len(l_list1)



while(i<k):

    print(l_list1[i])

    i=i+1
l_list2 = [1,15,47,656,100,245,70,-331,66]



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



print ("Maximum sayı : " , v_max)

print ("Minimum sayı : " , v_min)

b=v_max

v_loopindex=0

for b in l_list2[5:9]:

    print(b,v_loopindex,'. sayıdır')

    v_loopindex=v_loopindex+1
