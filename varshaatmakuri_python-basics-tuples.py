#Tuples are same as list but immutable
a=(1,2,3,4);
print('Printing a');
print(a);
a=a+(5,6);
print('Printing a');
print(a);
a=a*3;
print('Printing a');
print(a);
print(len(a));
del(a);
a=(1,2,3,4,5);
print(2 in a);
print(0 in a);
for i in a:
    print(i);




