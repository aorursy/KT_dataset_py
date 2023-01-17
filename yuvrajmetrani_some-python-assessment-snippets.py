arr = [2,3,4,3,2,1]

n = 3



def get_repeats(arr, n):

    repeats = 0

    for element in arr:

        if(n == element):

            repeats += 1

    return (repeats)



print(get_repeats(arr,n))
a = {1:9, 2:8, 3:7, 4:6, 5:5}

a.items()
b = {1:9, 2:8, 3:7, 4:6, 5:5}

b.get(6)
try:

    [1,2,3][4]

except IndexError:

    print("error")

finally:

    print("cleaning up")
a = 7

a.__str__()
set([1,2,1]) == set([1,2])
def f():

    f()

    return(42)



f()
class test():

    id=0

    def __init__(self,id):

        self.id = id

        id = 2



t = test(1)

t.id
m=re.search(r'(ab[cd])',"acdeabdabcde")

m.groups()




x1,y1 = 2,3

x2,y2 = 3,3

x3,y3 = 4,4



import math



p1 = (x1, y1)

p2 = (x2, y2)

p3 = (x3, y3)



def distanct_2points(a, b):

    #a and b are points with coordintes (x,y)

    #distanct between a and b is sqrt

    x1, y1, x2, y2 = a[0], a[1], b[0], b[1]

    dist=math.sqrt((x2-x1)**2+(y2-y1)**2)

    return dist





#average distance between 3 points

avg_dist = (distanct_2points(p1,p2) + distanct_2points(p2,p3) + distanct_2points(p3,p1))/3



print(f"Average distance between 3 points : { avg_dist } .")

    
def isPalindrome(s):

    return s == s[::-1]



# Driver code

s = "malwyalam"

ans = isPalindrome(s)

 

if ans:

    print("Yes")

else:

    print("No")