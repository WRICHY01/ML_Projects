listx = [2, "tall", [4, 3, 4], [5, 1]]
listy = listx[:]

print(f"{'--' * 15} Before_Modification {'--' * 15}")
print(f"listx value: {listx}, and its memory address is {hex(id(listx))}")
print(f"listx first element value: {listx[0]}, and its memory address is {hex(id(listx[0]))}")
print(f"listx second element value: {listx[1]}, and its memory address  is {hex(id(listx[1]))}")
print(f"listx third element value: {listx[2]}, and its memory address is {hex(id(listx[2]))}")
print(f"listx fourth element value: {listx[3]}, and its memory address is {hex(id(listx[3]))}")
print(f"listy value: {listy}, and its memory address is {hex(id(listy))}")
print(f"listy first element value: {listy[0]}, and its memory address is {hex(id(listy[0]))}")
print(f"listy second element value: {listy[1]}, and its memory address  is {hex(id(listy[1]))}")
print(f"listy third element value: {listy[2]}, and its memory address is {hex(id(listy[2]))}")
print(f"listy fourth element value: {listy[3]}, and its memory address is {hex(id(listy[3]))}")

# listx.remove("tall")
listx[1] = "very_tall"
listx[2].append(14)
listx[2][2] = 5

print(f"{'--' * 15} After_Modification {'--' * 15}")
print(f"listx value: {listx}, and its memory address is {hex(id(listx))}")
print(f"listx first element value: {listx[0]}, and its memory address is {hex(id(listx[0]))}")
print(f"listx second element value: {listx[1]}, and its memory address  is {hex(id(listx[1]))}")
print(f"listx third element value: {listx[2]}, and its memory address is {hex(id(listx[2]))}")
print(f"listx fourth element value: {listx[3]}, and its memory address is {hex(id(listx[3]))}")
print(f"listy value: {listy}, and its memory address is {hex(id(listy))}")
print(f"listy first element value: {listy[0]}, and its memory address is {hex(id(listy[0]))}")
print(f"listy second element value: {listy[1]}, and its memory address  is {hex(id(listy[1]))}")
print(f"listy third element value: {listy[2]}, and its memory address is {hex(id(listy[2]))}")
print(f"listy fourth element value: {listy[3]}, and its memory address is {hex(id(listy[3]))}")