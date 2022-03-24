import numpy as np
import pandas as pd
import os
import sys, pprint

class A:
    XY = 5

    class B:
        def __init__(self):
            self.bx = np.empty(shape=self.XY)
            self.by = np.empty(shape=self.XY)

    class C:
        def __init__(self):
            self.dt = np.dtype([('bx', np.float64, (self.XY, )), ('cx', np.float64, (self.XY, ))])
            self.cx = np.empty(shape=self.XY, dtype=self.dt)

    def __init__(self):
        self.C.__init__(self)


if __name__ == '__main__':
    pprint.pprint(sys.path)
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