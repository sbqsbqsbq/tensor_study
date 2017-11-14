import random

a = [];
b_list = [];

for i in range (1, 47):
    a.append(i)

for i in range (0, 10):
    b = random.sample(a, 6)
    b_list.append(b)

for i in range (0, 10) :
    print(str(i+1) + "번째")
    print(b_list[i])