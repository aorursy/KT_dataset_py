#first, let user input varibles x,y,z
x=int(input('Please give a integer value to variable x:'))
y=int(input('Please give a integer value to variable y:'))
z=int(input('Please give a integer value to variable z:'))
alist=[x,y,z] 
blist=[]
clist=[]
for i in alist:               #test the varible one by one in alist
    if i>=0:                  #consider postive and negative two different situations
        if (1+i)%2==0:        #test whether it's odd
            blist.append(i)   #add odd varibles to blist
    if i<0:                   #negative situation
        if (1+abs(i))%2==0:
            blist.append(i)    
if blist==clist:              #there is no odd varible, so blist is empty
    print('None of them are odd.') 
else:
    print(max(blist))         
numXs=int(input('How many times do you want to print the letter X:'))
x=0
while x<numXs:
    print('X',end='')
    x+=1