## Page 284/285

# Creating the class definition
class MyClass:
    MyVar = 0

MyInstance = MyClass()
MyInstance.MyVar

MyInstance.__class__
## Page 286

# Considering the built-in class attributes
print(dir(MyInstance))

help('__class__')
## Page 287

# Working with methods
class MyClass:
    def SayHello():
        print('Hello there!')
        
MyClass.SayHello()
## Page 288

# Creating instance methods - An instance method is one that is part of the individual instances
class MyClass:
    def SayHello(self): #The self argument points at the particular instance that the application is using to manipulate data.
        print("Hello there!")

MyInstance = MyClass()
MyInstance.SayHello()




## Page 289/290

## Working with constructors
# A constructor is a special kind of method that Python calls when it instantiates...
# an object by using the definitions found in your class.

# Python relies on the constructor to perform tasks such as initializing...
# (assigning values to) any instance variables that the object will need when it starts.

class MyClass:
    Greeting = ""
    def __init__(self, Name="there"):
        self.Greeting = Name + "!"
    def SayHello(self):
        print("Hello {0}" .format(self.Greeting))
        
MyInstance = MyClass()
MyInstance.SayHello()

MyInstance2 = MyClass("Amy")
MyInstance2.SayHello()

MyInstance.Greeting = "Harry!"
MyInstance.SayHello()
## Page 291/292

## Working with variables

# Class Variable

class MyClass:
    Greeting = ""
    def SayHello(self):
        print("Hello {0}".format(self.Greeting))

MyClass.Greeting = "Zelda"
MyClass.Greeting

MyInstance = MyClass()
MyInstance.SayHello()

# Instance Variable

class MyClass:
    def DoAdd(self, Value1=0, Value2=0):
        Sum = Value1 + Value2
        print("The sum of {0} plus {1} is {2}" .format(Value1, Value2, Sum))
        
MyInstance = MyClass()
MyInstance.DoAdd(1, 4)
## Page 293

# Using methods with variable argument lists
class MyClass:
    def PrintList1(*args):
        for Count, Item in enumerate(args):
            print("{0}. {1}".format(Count, Item))
    def PrintList2(**kwargs):
        for Name, Value in kwargs.items():
            print("{0} likes {1}".format(Name, Value))

MyClass.PrintList1("Red", "Blue", "Green")
MyClass.PrintList2(George="Red", Sue="Blue",Zarah="Green")
## Page 294/295

# Overloading Operators
class MyClass:
    def __init__(self, *args):
        self.Input = args
    def __add__(self, Other):
        Output = MyClass()
        Output.Input = self.Input + Other.Input
        return Output
    def __str__(self):
        Output = ""
        for Item in self.Input:
            Output += Item
            Output += " "
        return Output

Value1 = MyClass("Red", "Green", "Blue")
Value2 = MyClass("Yellow", "Purple", "Cyan")
Value3 = Value1 + Value2

print("{0} + {1} = {2}" .format(Value1, Value2, Value3))
## Page 296

## Creating a Class
class MyClass:
    def __init__(self, Name='Sam', Age=32):
        self.Name = Name
        self.Age = Age
    def GetName(self):
        return self.Name
    def SetName(self, Name):
        self.Name = Name
    def GetAge(self):
        return self.Age
    def SetAge(self, Age):
        self.Age = Age
    def __str__(self):
        return "{0} is aged {1}.".format(self.Name, self.Age)
    

## Page 298

## Extending Classes to Make New Classes

# Building the child class
class Animal:
    def __init__(self, Name="", Age=0, Type=""):
        self.Name = Name
        self.Age = Age
        self.Type = Type
    
    def GetName(self):
        return self.Name
    def SetName(self, Name):
        self.Name = Name
    def GetAge(self):
        return self.Age
    def SetAge(self, Age):
        self.Age = Age
    def GetType(self):
        return self.Type
    def SetType(self, Type):
        self.Type = Type
    def __str__(self):
        return "{0} is a {1} aged {2}".format(self.Name, self.Type, self.Age)

class Chicken(Animal):
    def __init__(self, Name="", Age=0):
        self.Name = Name
        self.Age = Age
        self.Type = "Chicken"
    def SetType(self, Type):
        print("Sorry, {0} will always be a {1}" .format(self.Name, self.Type))
    def MakeSound(self):
        print("{0} says Cluck, Cluck, Cluck!".format(self.Name))
        

## Pages 301/302

# Testing the class in an application
MyChicken = Chicken("Sally", 2)
print(MyChicken)

MyChicken.SetAge(MyChicken.GetAge() + 1)
print(MyChicken)

MyChicken.SetType("Gorilla")
print(MyChicken)

MyChicken.MakeSound()


