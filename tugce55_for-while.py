for n_numbers in range(1,16):

    print(n_numbers,'yasındasın')
v_m = '16 ya gireceğim.'

print(v_m)
for v_c in v_m:

    print(v_c)

    print("******")
for v_c in v_m.split():

    print(v_c)
l_list1=[1,5,34,7,54]

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
i = 0

while(i < 4):

    if i==0:

        print('misafir gelmedi')

    else:

        print( i,'. misafir geldi')

     

    i = i+1
print(l_list1)

print()



i = 0

k = len(l_list1)



while(i<k):

    print(l_list1[i])

    i=i+1

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



print ("Maximum sayı : " , v_max)

print ("Minimum sayı : " , v_min)
l_list3 = [-2,-5,-3,-45]



v_min = 0

v_max = 0



v_index = 0

v_len = len(l_list3)



while (v_index < v_len):

    v_current = l_list3[v_index]

    

    if v_current > v_max:

        v_max = v_current

    

    if v_current < v_min:

        v_min = v_current

    

    v_index = v_index+1



print ("Maximum sayı : " , v_max)

print ("Minimum sayı : " , v_min)
l_list4 = [3,5,7,6,100,255,71,34,85]



v_min = 0

v_max = 0



v_index = 0

v_len = len(l_list4)



while (v_index < v_len):

    v_current = l_list4[v_index]

    

    if v_current > v_max:

        v_max = v_current

    

    if v_current < v_min:

        v_min = v_current

    

    v_index = v_index+1



print ("Maximum sayı : " , v_max)

print ("Minimum sayı : " , v_min)
l_list2 = [3,5,7,-6,-100,-85,71,34,255]



v_min = 0

v_max = 0

v_maxindex=0

v_index = 0

v_len = len(l_list2)



while (v_index < v_len):

    v_current = l_list2[v_index]

    

    if v_current > v_max:

        v_max = v_current

        v_maxindex=v_index

    

    if v_current < v_min:

        v_min = v_current

        v_minindex=v_index

        

    

    v_index = v_index+1



print (v_index,'th number is maximum number and its : ', v_max ,)

print (v_minindex, 'th number is number and it is : ', v_min)



b=v_max

v_loopindex=0

for b in l_list2:

        print(b,v_loopindex,'. sayıdır.')

        v_loopindex=v_loopindex + 1

        v_loopindex=v_index

enbuyuk=0

for herbiri in l_list2:

    if enbuyuk>herbiri:

        enbuyuk=herbiri

        print('su anki en kucuk',enbuyuk)

son=enbuyuk

v_loopindex2=0

for son in l_list2:

    print(son,v_loopindex2,'. sayıdır')

    v_loopindex2=v_loopindex2+1

    
b=v_max

v_loopindex=0

for b in l_list2[8:9]:

    print(b,v_loopindex,'. sayıdır')

    v_loopindex=v_loopindex+1
