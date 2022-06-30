import time
from copy import deepcopy

import numpy as np
from numba import njit

from fileReader import Reader
from simpleaipack.search import SearchProblem
from structClasses import *
from numba_accelerated_funcs import partial_eval_jitted_one, partial_eval_jitted_two, actions_jitted, scalarize_jitted,\
                                    update_newState_solVec_jitted
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
    4) greedy: value.
    5) random walk: actions, result, value.
    6) random search: None.

"""


# spec = [
#     ('availableStatesSize', numba.int64),
#     ('algoName', numba.string),
#     ('algoSeed', numba.string),
#     ('problemSeed', numba.string),
#     ('numOfVarChangesInNeighborhood', numba.int64),
#     ('path', numba.string),
#     ('loadProblemFromFile', numba.boolean),
#     ('initialSolution', SolutionVector.class_type.instance_type),
#     ('reader', Reader.class_type.instance_type),
#     ('valuesPerVariables', ValuesPerVars.class_type.instance_type),
#     ('binaryConstraintsMatrix', numba.int64[:]),
#     ('maxValuesNum', numba.int64),
#     ('Ms', M.class_type.instance_type[:]),
# ]
#
# @jitclass(spec)
class COP(SearchProblem):
    def __init__(self, problemSeed, numOfVarChangesInNeighborhood, path, algoName, algoSeed, initialSolution=None,
                 loadProblemFromFile=True):
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

        # if not loadProblemFromFile:
        #     self.valuesPerVariables = ValuesPerVars()
        #     self.binaryConstraintsMatrix = np.zeros(shape=MAX_TOTAL_VALUES * MAX_TOTAL_VALUES, dtype=np.int64)
        #     self.maxValuesNum = None
        #     self.Ms = [M() for _ in range(MAX_NUM_OF_MS)]
        #     self.valuesPerVariablesInit()
        # else:
        self.reader = Reader(self.path, self.problemSeed)
        self.valuesPerVariables = self.reader.valuesPerVariable
        padding_with_zeroes = MAX_TOTAL_VALUES * MAX_TOTAL_VALUES - len(self.reader.binaryConstraintsMatrix)
        self.binaryConstraintsMatrix = np.pad(self.reader.binaryConstraintsMatrix, (0, padding_with_zeroes))
        self.maxValuesNum = self.reader.maxValuesNum
        self.Ms = self.reader.MS

        super().__init__(initial_state=self.initialSolution)

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
                        x = var1 * self.maxValuesNum + val1
                        y = var2 * self.maxValuesNum + val2
                        self.binaryConstraintsMatrix[x * MAX_TOTAL_VALUES + y] = np.random.randint(0, constraintsRatio)

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
            currVal, currM = partial_eval_jitted_one(currSolVar,
                                                     solutionVec.solutionVector,
                                                     self.valuesPerVariables.varsData[currSolVar].valuesM)

            partial_eval_jitted_two(currIsLegal, currSolVar, self.maxValuesNum, self.binaryConstraintsMatrix,
                                    currVal,
                                    outputEvaluation.gradesVector, solutionVec.solutionVector,
                                    currM, MsUsage[currM],
                                    self.valuesPerVariables.varsData[currSolVar].ucPrio,
                                    self.valuesPerVariables.varsData[currSolVar].valuesB,
                                    self.valuesPerVariables.varsData[currSolVar].valuesQ,
                                    self.valuesPerVariables.varsData[currSolVar].valuesP)

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
        actions = actions_jitted(self.availableStatesSize,
                                 self.valuesPerVariables.validVarAmount,
                                 self.numOfVarChangesInNeighborhood,
                                 self.valuesPerVariables.extract_valuesAmount())
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
        newState.solutionVector = update_newState_solVec_jitted(np.array(action), newState.solutionVector)
        return newState

    def value(self, state):
        """Evaluating the current state.

        Args:
            state (SolutionVector): current state .

        Returns:
            float: scalarized GradesVector resulted from applying self.evaluateSolution on input state.
        """
        evaluation = self.evaluateSolution(state)
        scalar = scalarize_jitted(evaluation.gradesVector, evaluation.valuesRange)
        return scalar

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

