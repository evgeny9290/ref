import numpy as np

x = np.array([[1,2,3],[4,5,6],[7,8,9]])
y = np.ones((3,3))
z1 = np.sum((x==1) * 1)
# print(z1)
z2 = np.sum((y == 1) * 1)
# print(np.subtract(z1,z2) )
# x[1,1] = 1
# x[2,2] = 1
# print(x)
# print(y)
# print(np.concatenate((list(np.argwhere(x==1)), list(np.argwhere(y==1)))))
# print(np.where(x[1:] == 1))
# print(list(zip(*np.argwhere(x==1))))
# print(x[0,2])