#doubly linked List 
class node():
    def __init__(self,data):
        self.data=data
        self.prev=None
        self.next=None
        
class dlinkedlist():
    def __init__(self):
        self.head=None

    def insertTail(self,data):
        newNode=node(data)
        if self.head==None:
            self.head=newNode
        else:
            temp=self.head
            while temp.next!=None:
                temp=temp.next
            temp.next=newNode
            newNode.prev=temp
            
    def printElements(self):
        temp=self.head
        while temp!=None:
            print(temp.data,end=' ')
            temp=temp.next
            #print()

            
    def insertHead(self,data):
        newNode=node(data)
        if self.head==None:
            self.head=newNode
        else:
            newNode.next=self.head#
            self.head.prev=newNode
            self.head=newNode
            


    def deleteByValue(self,value):
        if self.head==None:
            print('Empty list')
            
        else:
            prev=None
            curr=self.head
            #Assume the Number to delete is in the list
            while curr!=value:
                curr=curr.next
            #delete element
            temp=curr.next
            prev1=curr.prev
            #assign prev1 curr temp
#             temp=prev1.next
            prev1=temp.prev
            temp=prev1.next
        
    def reversel(self):
        if self.head==None:
            print('Empty doubly linked list')
        else:
            prev=None#1 
            curr=self.head#2
            while curr!=None:
                temp=curr.next#3
                curr.next=prev#2.next=1
                prev=curr#1=2
                curr=temp#2=3
            self.head=prev#self.head=1-None
L=dlinkedlist()
L.insertTail(1)
L.insertTail(2)
L.insertTail(3)
L.insertTail(4)
L.insertTail(5)
L.insertTail(6)
L.printElements()
L.insertHead(4)
L.printElements()
L.reversel()
L.printElements()
L.deleteByValue(4)
L.printElements()