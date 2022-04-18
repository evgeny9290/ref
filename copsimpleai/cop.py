from simpleaipack.search import SearchProblem
from fileReader import Reader
from structClasses import *

import numpy as np
from copy import deepcopy

"""
COP class inherits from SimpleAi SearchProblem module.
Responsible for:
    i) self problem creation if needed and can read problem from existing created COP problem.
    ii) executing localsearch algorithms from SimpleAi package implementing needed functions for the algorithms from:
        SearchProblem.
        
functions req for algorithms:
    1) simulated_annealing: actions, result, value.
    2) hill_climbing_stochastic: actions, result, value.
    3) beam_best_first: actions, result, value, generate_random_state.
"""


class COP(SearchProblem):
    def __init__(self, problemSeed, numOfVarChangesInNeighborhood, path, algoName, algoSeed, initialSolution=None, loadProblemFromFile=True):
        """COP problem class. used to initialize COP problem and solve it using various local search algorithms.

        Args:
            problemSeed (int): COP problem seed number.
            numOfVarChangesInNeighborhood (int): size of the neighborhood of a solution.
            path (str): path to the folder which contains all the files needed to initialize a COP problem created.
            algoName (str): name of an algorithm.
            algoSeed (str): stochastic behaviour of algorithms.
            initialSolution (SolutionVector, optional): leave empty if want to start from init SolutionVector
                else provide SolutionVector.
            loadProblemFromFile (bool, optional): True if initialize COP problem from created COP problem
                else False for self creation on COP problem.

        Returns:
             None.

        Notes:
            self.availableStatesSize: is set to be 50, purpose is to not search all possible delta states of current
                SolutionVector state but only generate 50 of them.
            can be changed to increase performance but run slower.
        """
        self.availableStatesSize = 1
        self.algoName = algoName
        self.algoSeed = algoSeed
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
        """Calculating the corresponding location for 1d array

        Args:
            x (int): row index
            y (int): col index

        Returns:
             int: location inside 1d array.
        """
        return x * MAX_TOTAL_VALUES + y

    def valuesPerVariablesInit(self):
        """COP self initialization problem. python implementation for COP constructor from localsearch.h it cpp.

        Args:
            None.

        Returns:
             None.
        """
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
        """Generating new SolutionVector randomized in numOfVarChangesInNeighborhood places from currentSolution.

        Args:
            currentSolution (SolutionVector): current SolutionVector.

        Returns:
            SolutionVector: randomized SolutionVector in numOfVarChangesInNeighborhood places.
        """
        outputSolution = SolutionVector()
        outputSolution.solutionVector = deepcopy(currentSolution.solutionVector)

        for var in range(1, self.numOfVarChangesInNeighborhood + 1):
            randIntForVar = np.random.randint(0, self.valuesPerVariables.validVarAmount)
            randIntForVal = np.random.randint(0, self.valuesPerVariables.varsData[randIntForVar].valuesAmount)
            outputSolution.solutionVector[randIntForVar] = randIntForVal

        return outputSolution

    def evaluateSolution(self, solutionVec):
        """Evaluating the passed SolutionVector as GradesVector.

        Args:
            solutionVec (SolutionVector): current SolutionVector.
        Returns:
            GradesVector: evaluation of passed SolutionVector.
        """
        outputEvaluation = GradesVector()
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

            if currIsLegal:  # was -= for minimization. testing +=
                MsUsage[currM].amount -= 1
                outputEvaluation.gradesVector[2 * currPrio] -= 1
                outputEvaluation.gradesVector[2 * currPrio + 1] -= currP
                outputEvaluation.gradesVector[LEVEL_OF_B] -= currB
                outputEvaluation.gradesVector[LEVEL_OF_Q] -= currQ

        return outputEvaluation

    def actions(self, state):
        """Generates self.availableStatesSize random 2-tuples.
        what index to change and into what number. because it is random it doesn't matter what is the current state.

        Args:
            state (SolutionVector): current state.

        Returns:
            list[list[(int, int)]]: a single action is performing self.numOfVarChangesInNeighborhood in current state.
                and there are self.availableStatesSize such actions.
        """
        actions = []
        for action in range(self.availableStatesSize):  # change (rand_idx, rand_val)
            change = []
            for var in range(1, self.numOfVarChangesInNeighborhood + 1):
                rand_entry = np.random.randint(0, self.valuesPerVariables.validVarAmount)
                rand_val = np.random.randint(0, self.valuesPerVariables.varsData[rand_entry].valuesAmount)
                change.append((rand_entry, rand_val))
            actions.append(change)
        return actions

    def result(self, state, action):
        """Applying action to the current state. meaning applying delta changed which are located in action.

        Args:
            state (SolutionVector): current SolutionVector.
            action (list[(int, int)]): what index to perform the change on, the change.

        Returns:
            SolutionVector: new state after performing an action on it.
        """
        newState = deepcopy(state)
        for idx, val in action:
            newState.solutionVector[idx] = val
        return newState

    def value(self, state):
        """Evaluating the current state.

        Args:
            state (SolutionVector): current state .

        Returns:
            float: scalarized GradesVector resulted from applying self.evaluateSolution on input state.
        """
        evaluation = self.evaluateSolution(state)
        return evaluation.scalarize()

    def grade_value(self, state):
        return self.evaluateSolution(state)

    def generate_random_state(self):
        """Generate random starting points from self.initialSolution as SolutionVector's

        Args:
            None.

        Returns:
            SolutionVector: random SolutionVector.
        """
        return self.generateSingleNeighbor(self.initialSolution)


# if __name__ == '__main__':
#     problem = COP(problemSeed=3118,
#                   numOfVarChangesInNeighborhood=5,
#                   path=r'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\CPP_Problems\\',
#                   algoName='GREEDY',
#                   algoSeed='331991908',
#                   initialSolution=None,
#                   loadProblemFromFile=True)
#
#     x = [84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84,
#          84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84,
#          84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#
#     y = [82, 0, 0, 78, 44, 0, 0, 18, 9, 2, 0, 1, 0, 57, 67, 0, 28, 37, 3, 16, 0, 21, 64, 30, 48, 0, 11, 40, 64, 68, 17,
#          39, 10, 66, 82, 7, 80, 23, 73, 15, 16, 22, 82, 61, 27, 46, 73, 67, 58, 68, 41, 44, 73, 71, 46, 77, 50, 58, 27,
#          81, 23, 51, 14, 38, 5, 33, 4, 74, 21, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#
#     sv = SolutionVector()
#     sv.solutionVector = x
#     print(problem.evaluateSolution(sv))
