# import numpy as np
# import pandas as pd
# import os
# import sys, pprint
# import math
#
#
# class A:
#     XY = 5
#
#     class B:
#         def __init__(self):
#             self.bx = np.empty(shape=self.XY)
#             self.by = np.empty(shape=self.XY)
#
#     class C:
#         def __init__(self):
#             self.dt = np.dtype([('bx', np.float64, (self.XY, )), ('cx', np.float64, (self.XY, ))])
#             self.cx = np.empty(shape=self.XY, dtype=self.dt)
#
#     def __init__(self):
#         self.C.__init__(self)
#
#
# def x():
#     return 1,2

def func(A, L, K):
    for i in range(1, len(A)):
        A[i] += A[i-1]

    res = A[L + K - 1]
    maxL = A[L - 1]
    maxM = A[K - 1]

    for i in range(K+L, len(A)):
        maxL = max(maxL, A[i - K] - A[i - K - L])
        maxK = max(maxM, A[i - L] - A[i - L - K])
        res = max(res, maxL + (A[i] - A[i - K]), maxK + (A[i] - A[i - L]))

    return res


if __name__ == '__main__':
    A = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
    K = 3
    L = 2
    print(func(A, L, K))




#     print(type(x()))
#     if type(x()) is tuple:
#         print("asasassasasaas")

    # x = ["asdasda_11", "asdasdaaaaaa_22"]
    # res = []
    # for _ in x:
    #     r = []
    #     for i in range(4):
    #         r.append(i)
    #     res.append(r)
    # print(res)
    # print(math.exp(-91234123912312 / 0.00000000000001))
    # pprint.pprint(sys.path)
    # x = A()
    # print(x.cx['bx'])
    # path = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\Problems\\'
    # files = os.listdir(path)
    # print(files)
    # first_bin_max = files[0]
    # print(first_bin_max)
    #
    # with open(path + first_bin_max) as f:
    #     x = np.array(f.read().split(), dtype=np.int16)
    #
    # print(type(x), x.shape)
    # print(x[:20], type(x[2]), x[1], x[12312])



    # x = np.array([3,4,4,1,1,4,0,4,3,0], dtype=np.int16)
    # padded = np.pad(x,(0,5))
    # print(padded)
    # def __init__(self, ucPrion, valuesAmount, valuedVarAmount=1):
        # self.VarData.__init__(self, ucPrion, valuesAmount)
        # self.dt = np.dtype([('ucPrion', np.int32, ),
        #                     ('valuesQ', np.int32, (self.MAX_VALUES_OF_VAR, )),
        #                     ('valuesB', bool, (self.MAX_VALUES_OF_VAR, )),
        #                     ('valuesP', np.int32, (self.MAX_VALUES_OF_VAR, )),
        #                     ('valuesM', np.int32, (self.MAX_VALUES_OF_VAR, )),
        #                     ('valuesAmount', np.int64, )])

        # self.varsData = np.empty(shape=self.MAX_NUM_OF_VARS, dtype=self.dt)
        # self.validVarAmount = None
        # self.varsData = [self.VarData(None, None)] * MAX_NUM_OF_VARS