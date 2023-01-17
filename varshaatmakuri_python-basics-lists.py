# Print Statement
print('Hello!');
#Variables
a=10;
print(a);
print('----Lists-------');
b=[1,2,3,4,5,6,7,8,9,10];
print(b);

print('Index starts with 0.');
print(b[0]);

print('Here b and c point to the same memory location');
c=b;
c[0]=11;
print(b);
print(c);



print('Accessing elements');

#Starts with index 1.
print(b[1:]);
#Starts with 1 and ends with 2(3-1).
print(b[1:3]);
#Prints alternate element
print(b[::2]);
#Reverse Order
print(b[::-1]);

print('Updating Lists');
a=[1,2,3,4,5,6,7,8,9,10];
print('Printing a');
print(a);
print('Updating  element at index 2 to 1234');
a[2]=1234;
print(a);

print('Deleting items from list');
print('Printing a');
print(a);
print('Deleting element at index 4');
del(a[4]);
print(a);
print('Length of List');
a=[1,2,3,4,5];
print('Printing a');
print(a);
print('Length is %s'%len(a));
print('Concatenating 2 lists');
a=[1,2,3];
b=[2,3,4];

c=a+b;
print(c);

print('APPEND,POP');

a=[1,2,3,4,5];
print('Printing a');
print(a);

print('APPEND:');
print('Appending 11');
a.append(11);
print('Printing a');
print(a);
print('POP');
a.pop();
print('Printing a');
print(a);
print('POP at index 2');
a.pop(2);
print('Printing a');
print(a);


print('EXTEND, INSERT');
a=[1,2,3,4,5];
print('Printing a');
print(a);
print('Extending list by adding 22 and 33');
a.extend([22,33]);
print('Printing a');
print(a);
print('Inserting 333 at index 4');
a.insert(4,333);
print('Printing a');
print(a);
print("REMOVE, INDEX");
a=[1,3,2,4,2,1];
print('Printing a');
print(a);

print('remove 1');
#Removes 1st occurrence
a.remove(1);

print('Printing a');
print(a);

print("Index of element 2");
print(a.index(2));





print("CLEAR LIST");
print('Printing a');
print(a);
print("CLEARING");
a.clear();
print('Printing a');
print(a);

print("SORT");
a=[3,1,5,7,2];
print('Printing a');
print(a);
a.sort();
print('AFTER SORTING');
print('Printing a');
print(a);
print("REVERSE SORTING");
a.sort(reverse = True);
print('Printing a');
print(a);





