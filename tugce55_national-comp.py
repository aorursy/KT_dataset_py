d_company={'baykar':'bayraktar_tb2','stm':'kargu','tei':'hürkuş'}



print(d_company)

print(type(d_company))
v_iha=d_company['baykar']



print(v_iha)

print(type(v_iha))



v_suruiha=d_company['stm']

print(v_suruiha)

print(type(v_suruiha))
d_envanter={'baykar':70,'tei':35,'bmc':200}

print(d_envanter)

v_1=d_envanter['baykar']

print(v_1)
v_keys=d_company.keys()

v_values=d_company.values()

v_k2=d_envanter.keys()

v_v2=d_envanter.values()





print(v_keys)

print(v_values)

print(v_k2)

print(v_v2)
v_baykar = 35

v_tei = 35



if v_baykar > v_tei:

    print('baykar, tei den daha fazla ülkeye hizmet ediyor')

elif v_baykar < v_tei:

    print('tei, baykar dan daha fazla ülkeye hizmet ediyor')

else :

    print("Ülkeye katkı kıyaslanamaz")
def f_envanter(v_Comp1 , v_Comp2):

    if v_Comp1 > v_Comp2:

        print(v_Comp1 , " is greater then " , v_Comp2)

    elif v_Comp1 < v_Comp2:

        print(v_Comp1 , " is smaller then " , v_Comp2)

    else :

        print("These " , v_Comp1 , " variables are equal")

        

f_envanter(33,44)

f_envanter(66,22)

f_envanter(11,11)
def f_IncludeOrNot(v_search, v_searchList):

    if v_search in v_searchList :

        print("Good news ! ",v_search , " is in list.")

    else :

        print(v_search , " is not in list. Sorry :(")



l_list = list(d_company.keys())

print(l_list)

print(type(l_list))



f_IncludeOrNot("baykar" , l_list)

f_IncludeOrNot("aselsan" , l_list)