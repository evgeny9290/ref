from Constants import *
import numpy as np
"""
same class names as struct named defined within COP class in localsearch.h file.
"""
# varDataSpec = [
#     ('ucPrio', numba.int64),
#     ('valuesQ', numba.int64[:]),
#     ('valuesB', numba.int64[:]),
#     ('valuesP', numba.int64[:]),
#     ('valuesM', numba.int64[:]),
#     ('valuesAmount', numba.int64)
# ]

# vpvSpec = [
#     ('validVarAmount', numba.int64),
#     ('varsData', VarData.class_type.instance_type[:])
# ]

#
# mSpec = [
#     ('amount', numba.int64)
# ]

# solVecSpec = [
#     ('solutionVector', int64[:])
# ]

# gradeVecSpec = [
#     ('valuesRange', numba.int64[:]),
#     ('gradesVector', numba.int64[:])
# ]

# @jitclass(spec=varDataSpec)
class VarData:
    def __init__(self):
        self.ucPrio = 0
        self.valuesQ = np.zeros(shape=MAX_VALUES_OF_VAR, dtype=np.int64)
        self.valuesB = np.zeros(shape=MAX_VALUES_OF_VAR, dtype=np.int64)
        self.valuesP = np.zeros(shape=MAX_VALUES_OF_VAR, dtype=np.int64)
        self.valuesM = np.zeros(shape=MAX_VALUES_OF_VAR, dtype=np.int64)
        self.valuesAmount = 1

    def __lt__(self, other):
        return self.ucPrio < other.ucPrio


# @jitclass(spec=vpvSpec)
class ValuesPerVars:
    def __init__(self):
        self.validVarAmount = 1
        self.varsData = np.array([VarData() for _ in range(MAX_NUM_OF_VARS)], dtype=object)

    def extract_valuesAmount(self):
        return np.array([x.valuesAmount for x in self.varsData], dtype=np.int64)


# @jitclass(spec=mSpec)
# class M:
#     def __init__(self):
#         self.amount = None
#
#     def __str__(self):
#         return f"M amount: {self.amount}"


# @jitclass(spec=solVecSpec)
class SolutionVector:
    def __init__(self):
        self.solutionVector = np.zeros(shape=MAX_NUM_OF_VARS, dtype=np.int64)

    def __str__(self):
        return f"solution vector: {self.solutionVector}"

    def __lt__(self, other):
        return any(self.solutionVector < other.solutionVector)

    def __ge__(self, other):
        return not self.__lt__(other)


# @jitclass(spec=gradeVecSpec)
class GradesVector:
    def __init__(self):
        self.valuesRange = self.init_values_range()
        self.gradesVector = np.zeros(shape=MAX_LENGTH_OF_GRADES_VECTOR, dtype=np.int64)

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
                             MAX_NUM_OF_VARS * 1], dtype=np.int64)
        return init_arr
