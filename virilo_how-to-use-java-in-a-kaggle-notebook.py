!ls -la /usr/lib/jvm/
!java -version
java_code='''

public class HelloWorld {

   public static void main(String[] args) {

      System.out.println("Hello, World");

   }

}

'''



f = open('HelloWorld.java','w')

f.write(java_code)

f.close()

!javac HelloWorld.java

!java HelloWorld