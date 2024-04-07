class GrandParent:
    def test(self):
        print("GrandParent's test()")

class Parent(GrandParent):
    def test(self):
        print("Parent's test()")

class Child(Parent):
    def test(self):
        super(Child, self).test()

c = Child()
c.test()