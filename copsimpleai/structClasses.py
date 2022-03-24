from Constants import *
import numpy as np


class VarData:
    def __init__(self):
        self.ucPrio = None
        self.valuesQ = np.zeros(shape=MAX_VALUES_OF_VAR, dtype=np.int16)
        self.valuesB = np.zeros(shape=MAX_VALUES_OF_VAR, dtype=np.int16)
        self.valuesP = np.zeros(shape=MAX_VALUES_OF_VAR, dtype=np.int16)
        self.valuesM = np.zeros(shape=MAX_VALUES_OF_VAR, dtype=np.int16)
        self.valuesAmount = None


class ValuesPerVars:
    def __init__(self):
        self.validVarAmount = 1
        self.varsData = [VarData()] * MAX_NUM_OF_VARS


class M:
    def __init__(self):
        self.amount = None


class SolutionVector:
    def __init__(self):
        self.solutionVector = np.zeros(shape=MAX_NUM_OF_VARS, dtype=np.int32)

    def __str__(self):
        return f"solution vector: {self.solutionVector}"


class GradesVector:
    def __init__(self):
        self.valuesRange = self.init_values_range()
        self.gradesVector = np.zeros(shape=MAX_LENGTH_OF_GRADES_VECTOR, dtype=np.int32)

    def scalarize(self):
        scalarizedVal = np.float64(0)
        currWeight = np.float64(1.0)
        for gradeIdx in range(MAX_LENGTH_OF_GRADES_VECTOR - 1, -1, -1):
            if self.valuesRange[gradeIdx] == 0:
                continue
            scalarizedVal += self.gradesVector[gradeIdx] * currWeight
            currWeight *= self.valuesRange[gradeIdx] + 1

        return abs(scalarizedVal)

    def __str__(self):
        return f"gradeVector : {self.gradesVector} \nscalarization : {self.scalarize()}"

    @staticmethod
    def init_values_range():
        init_arr = np.array([MAX_NUM_OF_VARS * 1,
                             MAX_NUM_OF_VARS * 10,
                             MAX_NUM_OF_VARS * 1,
                             MAX_NUM_OF_VARS * 10,
                             MAX_NUM_OF_VARS * 1,
                             MAX_NUM_OF_VARS * 10,
                             MAX_NUM_OF_VARS * 1,
                             MAX_NUM_OF_VARS * 10,
                             MAX_NUM_OF_VARS * 1,
                             MAX_NUM_OF_VARS * 10,
                             MAX_NUM_OF_VARS * 1,
                             MAX_NUM_OF_VARS * 10,
                             MAX_NUM_OF_VARS * 1,
                             MAX_NUM_OF_VARS * 10,
                             MAX_NUM_OF_VARS * 1,
                             MAX_NUM_OF_VARS * 10,
                             MAX_NUM_OF_VARS * 1,
                             MAX_CONSTRAINTS_RATIO * (MAX_NUM_OF_VARS * MAX_NUM_OF_VARS - MAX_NUM_OF_VARS) // 2,
                             MAX_NUM_OF_VARS * 1])
        return init_arr

