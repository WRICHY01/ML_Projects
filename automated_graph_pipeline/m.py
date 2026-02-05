l1 = [3, 2, 3]
str1 = 23
print("str1 id before: ", str1, type(str1), hex(id(str1)))
str1 = str(str1)
print("str1 id after: ", str1, type(str1), hex(id(str1)))

fruits = ["grapes", "guava","aubergine", "peaches", "cucumber", "mango", ["orange", "cherry"]]
fruits_list = fruits
fruits = tuple(fruits)
fruits_tup = fruits[:]

print(f"the tuple {fruits}, is in the memory address: {hex(id(fruits))}")
print(fruits_list)
print(f"the tuple copy: {fruits_tup}, is in the memory address: {hex(id(fruits_tup))}")


items = (1, 2, 5, 6)
tst = (x * x for x in items)

print(f"generator object is: {tst}")
print(f"memory address of tst is: {hex(id(tst))}")
print(f"memory address of items is: {hex(id(items))}")

tst = tuple(tst)
print(f"tuple object is {tst}")
print(f"memory address of tst is: {hex(id(tst))}")
