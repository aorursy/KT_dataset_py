list=input('input 10 integers, and separate it with space:') #input 10 integers
integers=list.split()        #use split to seperate the numbers
dlist=map(int,integers)      #use map to make the strings become integers
empty_list=[]            
even_list=[]
for i in dlist:
    if i%2==0:
        even_list.append(i)   #add even varibles to blist
        
if even_list==empty_list:
    print('No even number was entered.')
else:
    print(max(even_list))
integer=int(input('Please input an integer:'))
root=1
while root<integer:
    pwr=2
    while pwr<7 and root**pwr<=integer:
        if root**pwr==integer:
            break
        else:
            pwr+=1
    if root**pwr==integer:
        print('{} and {}'.format(root,pwr))
        break
    else:
        root+=1
if root==integer:
    print(' no such pair of integers exists ')
# Find the cube root of an integer, and we use if to seperately execute the positive condition and negetive condition
x = int(input('Enter an integer: '))
epsilon = float(input('Enter the acceptable error: '))
numGuesses = 0
if x>0:
    low = 0.0
    high = max (1.0, x)
    ans = (high + low)/2.0
    while abs(ans**3 - x) >= epsilon:
        print ('low =', low, 'high = ', high, 'ans =', ans)
        numGuesses += 1 
        if ans**3 < x:
            low = ans
        else:
            high = ans
        ans = (high + low)/2.0
elif x<0:
    low = min(x,-1)
    high = 0.0
    ans = (high + low)/2.0
    while abs(ans**3 - x) >= epsilon:
        print ('low =', low, 'high = ', high, 'ans =', ans)
        numGuesses += 1 
        if ans**3 < x:
            low= ans
        else:
            high= ans
        ans = (high + low)/2.0
print ('numGuesses =', numGuesses)
print(ans, 'is close to cube root of',x)