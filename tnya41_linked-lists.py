class node():
    def __init__(self,data,next):
        self.data=data
        self.next=None
    
class linkedList():
    def __init__(self):
        self.head=None
        
    #function to insert an element in a linked list
    def insertElement(self,data):
        newNode=node(data)
        temp=self.head
        if self.head==None:
            self.head=newNode
        else:
            while temp.next!=None:
                temp=temp.next
            #inserting newNode At last element
            temp=newNode
    ##########################################################    
    #function to insert element at a position 
    # if pos==1 2 cases -elf.head is none
    # self.head
    
    
#     def insertAtPos(self,data,pos):
#         newNode=node(data)
#         if pos==1:
#             if self.head==None:
#                 self.head=newNode
#             else:
#                 temp=self.head
#                 self.head=newNode
#                 newNode.next=temp
#         else:
#             temp=self.head
#             for i in range(pos-2):
#                 temp=temp.next
            
            
            
        
    #function to delete an element given its value
    def deleteValue(self,value):
        temp=self.head
        while temp.data==value:
            temp=temp.next
        temp.next= temp.next.next
        
        
            
    #######################################################
    # function to delete an element at a particular position
    
    
#     def deleteAtPos(self,pos):
#         if pos==1:
#             temp=self.head
            
    
    # Function to print the elements of a singly linked List
    # Call the function as L.printvalue()
    def printValue(self):
        temp=self.head
        while temp.next!=None:
            temp=temp.next
        print(temp.data)
    
    #Function to print the middle element of a linked list 
    # Time Complexity for 2 for loops is O(n^2)
    def printMid2(self):
        temp,temp2=self.head
        count=0
        while temp.next!=None:
            count+=1
            temp=temp.next
        n=count/2
        ##found count
        while n!=0:
            n-=1
            temp2=temp2.next
        print(temp2.data)
        
    
    # Function to print the the middle element of a linked list using one loop 
    def printMid1(self):
        temp1,temp2=self.head
        while temp1.next!=None:
            temp1=temp1.next.next
            temp2=temp2.next
        print(temp2.data)
        
class node():
    def __init__(self,data):
        self.data=data
        self.next=None
    
class linkedList():
    def __init__(self):
        self.head=None
    ###############################################
    #function to insert an element in a linked list
    ###############################################
    def insertElement(self,data):
        newNode=node(data)
        if self.head==None:
            self.head=newNode
        else:
            temp=self.head
            while temp.next!=None:
                temp=temp.next
            #inserting newNode At last element
            temp.next=newNode
            
    #######################################################
    # Function to print the elements of a singly linked List
    # Call the function as L.printvalue()
    #######################################################
    def printValue(self):
        temp=self.head
        while temp.next!=None:
            print(temp.data)
            temp=temp.next
            
    #####################################################        
    #Function to print the middle element of a linked list 
    # Time Complexity for 2 for loops is O(n^2)
    
    #####################################################
    def printMid2(self):
        temp,temp2=self.head
        count=0
        while temp.next!=None:
            count+=1
            temp=temp.next
        n=count/2
        ##found count
        while n!=0:
            n-=1
            temp2=temp2.next
        print(temp2.data)
        
    #######################################################
    # Function to print the the middle element of a linked list using one loop 
    ########################################################
    def printMid1(self):
        temp1=self.head
        temp2=self.head
        while temp1.next!=None:
            temp1=temp1.next.next
            temp2=temp2.next
        print(temp2.data)
        
    ##################################################
    #function to delete an element given its value
                
    ##################################################
    def deleteByValue(self,value):
        if value==self.head.data:
            self.head=self.head.next
        else:
            temp=self.head
            while temp.data==value:
                temp=temp.next
            temp.next= temp.next.next
        
L=linkedList()
L.insertElement(1)
L.insertElement(2)
L.insertElement(3)
L.insertElement(4)
L.insertElement(5)
L.insertElement(6)
L1=linkedList()
L1.insertElement(1)
L1.insertElement(2)
L1.insertElement(3)
L1.insertElement(4)
L1.insertElement(5)
L1.insertElement(6)
L1.insertElement(7)
L1.insertElement(8)
L1.insertElement(9)
L1.insertElement(10)
L1.printValue()
L.printValue()
print(L)
#L.printMid2()
L.deleteByValue(2)
L.printValue()
#L1.printMid1()