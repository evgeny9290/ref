from simpleaipack.search import SearchProblem
from simpleaipack.search.local import beam_best_first, hill_climbing_stochastic, simulated_annealing
from fileReader import Reader
from structClasses import *

import numpy as np
from copy import deepcopy
from time import time


class COP(SearchProblem):
    def __init__(self, problemSeed, numOfVarChangesInNeighborhood, path, initialSolution=None, loadProblemFromFile=True):
        self.availableStatesSize = 50
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
            padding_with_zeroes = MAX_TOTAL_VALUES * MAX_TOTAL_VALUES - len(self.reader.binaryConstraintsMatrix)
            self.binaryConstraintsMatrix = np.pad(self.reader.binaryConstraintsMatrix, (0, padding_with_zeroes))
            self.maxValuesNum = self.reader.maxValuesNum
            self.Ms = self.reader.MS

        super().__init__(initial_state=self.initialSolution)

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

    def actions(self, state):
        actions = []
        for action in range(self.availableStatesSize):  # change (rand_idx, rand_val)
            change = []
            for var in range(self.numOfVarChangesInNeighborhood):
                rand_entry = np.random.randint(0, self.valuesPerVariables.validVarAmount)
                rand_val = np.random.randint(0, self.valuesPerVariables.varsData[rand_entry].valuesAmount)
                change.append((rand_entry, rand_val))
            actions.append(change)

        return actions

    def result(self, state, action):
        newState = deepcopy(state)
        for idx, val in action:
            newState.solutionVector[idx] = val

        return newState

    def value(self, state):
        evaluation = self.evaluateSolution(state)
        return evaluation.scalarize()

    def generate_random_state(self):
        return self.generateSingleNeighbor(self.initialSolution)


if __name__ == '__main__':
    path = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\Problems\\'
    problem_seed = 500
    algo_seed = 100
    time_begining_init = time()
    copProblem = COP(problemSeed=problem_seed,
                     numOfVarChangesInNeighborhood=5,
                     path=path,
                     initialSolution=None,
                     loadProblemFromFile=True)

    print(f"init time: {time() - time_begining_init}")
    problem = copProblem
    # time_begining_init = time()
    # result = hill_climbing_stochastic(problem, iterations_limit=200, max_run_time=10, seed=algo_seed)
    time_begining_init = time()
    # result = simulated_annealing(problem, iterations_limit=1120, max_run_time=10, seed=algo_seed)
    result = beam_best_first(problem, beam_size=100, iterations_limit=2000, max_run_time=10, seed=algo_seed)
    print(f"algo run time: {time() - time_begining_init}")
    print(result)
    print(type(result))
    print(result.value)
    print(result.state)
    print(problem.evaluateSolution(result.state).scalarize())
