from simpleai.search import SearchProblem
from fileReader import Reader
from structClasses import *

import numpy as np
import pandas as pd
from copy import deepcopy
from time import time


class COP(SearchProblem):
    def __init__(self, problemSeed, numOfVarChangesInNeighborhood, path, initialSolution=None, loadProblemFromFile=True):
        self.problemSeed = problemSeed
        self.path = path
        self.numOfVarChangesInNeighborhood = numOfVarChangesInNeighborhood
        np.random.seed(problemSeed)
        self.initialSolution = SolutionVector() if initialSolution is None else initialSolution

        if not loadProblemFromFile:
            self.valuesPerVariables = ValuesPerVars()
            self.binaryConstraintsMatrix = np.zeros(shape=MAX_TOTAL_VALUES * MAX_TOTAL_VALUES, dtype=np.int16)
            self.maxValuesNum = None
            self.Ms = [M()] * MAX_NUM_OF_MS
            self.valuesPerVariablesInit()
        else:
            self.reader = Reader(self.path, self.problemSeed)
            self.valuesPerVariables = self.reader.valuesPerVariable
            self.binaryConstraintsMatrix = self.reader.binaryConstraintsMatrix
            self.maxValuesNum = self.reader.maxValuesNum
            self.Ms = self.reader.MS

        super().__init__(initial_state=self.initialSolution.solutionVector)

    @staticmethod
    def binConsIdx(x, y):
        return x * MAX_TOTAL_VALUES + y

    def valuesPerVariablesInit(self):
        # np.random.seed(self.problemSeed)
        constraintsRatio = np.random.randint(2, MAX_CONSTRAINTS_RATIO)
        varNum = np.random.randint(1, MAX_NUM_OF_VARS)
        self.maxValuesNum = min(MAX_TOTAL_VALUES // varNum, MAX_VALUES_OF_VAR)

        for m in range(MAX_NUM_OF_MS):
            self.Ms[m].amount = np.random.uniform(0, MAX_NUM_OF_R_PER_M)

        self.valuesPerVariables.validVarAmount = varNum
        for var in range(varNum):
            self.valuesPerVariables.varsData[var].valuesAmount = self.maxValuesNum
            self.valuesPerVariables.varsData[var].ucPrio = 0
            for val in range(self.maxValuesNum):
                self.valuesPerVariables.varsData[var].valuesB[val] = False if (np.random.randint(0, 1) == 0) else True
                self.valuesPerVariables.varsData[var].valuesM[val] = np.random.randint(0, MAX_NUM_OF_MS - 1)
                self.valuesPerVariables.varsData[var].valuesQ[val] = np.random.randint(1, NUM_OF_Q_VALUES)
                self.valuesPerVariables.varsData[var].valuesP[val] = np.random.randint(1, NUM_OF_P_VALUES)

        for var1 in range(self.valuesPerVariables.validVarAmount):
            for val1 in range(self.valuesPerVariables.varsData[var1].valuesAmount):
                for var2 in range(self.valuesPerVariables.validVarAmount):
                    for val2 in range(self.valuesPerVariables.varsData[var2].valuesAmount):
                        assert(var1 * self.maxValuesNum + val1 < MAX_TOTAL_VALUES and var2 * self.maxValuesNum + val2 < MAX_TOTAL_VALUES)
                        x = var1 * self.maxValuesNum + val1
                        y = var2 * self.maxValuesNum + val2
                        self.binaryConstraintsMatrix[self.binConsIdx(x, y)] = np.random.randint(0, constraintsRatio)

    def generateSingleNeighbor(self, currentSolution):
        outputSolution = SolutionVector()
        outputSolution.solutionVector = deepcopy(currentSolution.solutionVector)

        for var in range(1, self.numOfVarChangesInNeighborhood + 1):
            randIntForVar = np.random.randint(0, self.valuesPerVariables.validVarAmount)
            randIntForVal = np.random.randint(0, self.valuesPerVariables.varsData[randIntForVar].valuesAmount)
            outputSolution.solutionVector[randIntForVar] = randIntForVal

        return outputSolution

    def evaluateSolution(self, solutionVec):
        outputEvaluation = GradesVector()
        # outputEvaluation = np.zeros(shape=MAX_LENGTH_OF_GRADES_VECTOR, dtype=np.float32)
        MsUsage = deepcopy(self.Ms)
        for currSolVar in range(self.valuesPerVariables.validVarAmount):
            currIsLegal = True
            currVal = solutionVec.solutionVector[currSolVar]
            currM = self.valuesPerVariables.varsData[currSolVar].valuesM[currVal]

            if MsUsage[currM].amount == 0:
                continue

            currPrio = self.valuesPerVariables.varsData[currSolVar].ucPrio
            currB = self.valuesPerVariables.varsData[currSolVar].valuesB[currVal]
            currQ = self.valuesPerVariables.varsData[currSolVar].valuesQ[currVal]
            currP = self.valuesPerVariables.varsData[currSolVar].valuesP[currVal]

            for pastSolVar in range(currSolVar):
                pastVal = solutionVec.solutionVector[pastSolVar]
                assert(currSolVar*self.maxValuesNum + currVal <= MAX_TOTAL_VALUES and pastSolVar*self.maxValuesNum + pastVal <= MAX_TOTAL_VALUES)
                x = currSolVar*self.maxValuesNum + currVal
                y = pastSolVar*self.maxValuesNum + pastVal
                currBinaryVal = self.binaryConstraintsMatrix[self.binConsIdx(x, y)]
                if currBinaryVal == 0:
                    currIsLegal = False
                    break
                outputEvaluation.gradesVector[LEVEL_OF_BINARY_CONSTRAINTS] -= currBinaryVal

            if currIsLegal:
                MsUsage[currM].amount -= 1
                outputEvaluation.gradesVector[2 * currPrio] -= 1
                outputEvaluation.gradesVector[2 * currPrio + 1] -= currP
                outputEvaluation.gradesVector[LEVEL_OF_B] -= currB
                outputEvaluation.gradesVector[LEVEL_OF_Q] -= currQ

        return outputEvaluation


if __name__ == '__main__':
    path = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\Problems\\'
    problem_seed = 500
    time_begining_init = time()
    copProblem = COP(problemSeed=problem_seed,
                     numOfVarChangesInNeighborhood=5,
                     path=path,
                     initialSolution=None,
                     loadProblemFromFile=True)

    print(f"init time: {time() - time_begining_init}")

    rest_time = time()
    print(copProblem.valuesPerVariables.varsData[0].valuesQ)
    print()
    initialSolution = SolutionVector()
    print(initialSolution)
    print()
    generated_neighbor = copProblem.generateSingleNeighbor(initialSolution)
    print(generated_neighbor)
    print()
    evaluate_sol = copProblem.evaluateSolution(generated_neighbor)
    print(evaluate_sol)
    print()
    print(f"rest time: {time() - rest_time}")
    #
    # time_begining_init = time()
    # xxx = SolutionVector()
    # xxx.solutionVector = np.array([np.random.randint(1, 10) for _ in range(MAX_NUM_OF_VARS)])
    # print(xxx)
    # copProblem_fromInitial = COP(9, 3, xxx)
    # print(f"init time: {time() - time_begining_init}")


    # print(copProblem_fromInitial.Ms[7].amount)
    # print(copProblem_fromInitial.varsData[0].valuesQ[:5])
    # print(copProblem_fromInitial.varsData[0]['valuesQ'])
    # print(len(copProblem_fromInitial.varsData[0]))
    # print(len(copProblem_fromInitial.varsData[0]['valuesQ']))
    # print()
    # print(copProblem_fromInitial.varsData[0]['ucPrion'])