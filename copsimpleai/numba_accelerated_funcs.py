from copsimpleai.Constants import *
from numba import njit
import numpy as np
from typing import List, Tuple


@njit(cache=True, fastmath=True)
def scalarize_jitted(grade_vec: np.array, values_range: np.array):
    """Scalarize the GradesVector.

    Args:
        grade_vec (List[int]): Grades numpy 1d vector.
        values_range (List[int]): values range for the current GradesVector (numpy 1d vector).

    Returns:
        float: scalarization result of the current grade vector.
    """
    scalarizedVal = np.float64(0)
    currWeight = np.float64(1.0)
    for gradeIdx in range(MAX_LENGTH_OF_GRADES_VECTOR - 1, -1, -1):
        if values_range[gradeIdx]:
            scalarizedVal += grade_vec[gradeIdx] * currWeight
            currWeight *= values_range[gradeIdx] + 1

    return abs(scalarizedVal)


@njit(cache=True)
def actions_jitted(availableStatesSize: int, validVarAmount: int, neighborhoodSize: int, varsDataAmount: np.array):
    """Generates self.availableStatesSize random 2-tuples.
    what index to change and into what number. because it is random it doesn't matter what is the current state.

    Args:
        state (SolutionVector): current state.
        availableStatesSize (int): number of state actions to output
        validVarAmount (int): valid amount of variables
        neighborhoodSize(int): the number of changes req per solVector
        varsDataAmount(List[int]): valuesAmount available per Variable

    Returns:
        list[list[(int, int)]]: a single action is performing self.numOfVarChangesInNeighborhood in current state.
            and there are self.availableStatesSize such actions.
    """

    actions: List[List[Tuple[int, int]]] = [[(-1, -1)]]
    actions.clear()
    for action in range(availableStatesSize):  # change (rand_idx, rand_val)
        change: List[Tuple[int, int]] = [(-1, -1)]
        change.clear()
        for var in range(1, neighborhoodSize + 1):
            rand_entry = np.random.randint(0, validVarAmount)
            rand_val = np.random.randint(0, varsDataAmount[rand_entry])
            change.append((rand_entry, rand_val))
        actions.append(change)
    return actions


@njit(cache=True)
def partial_eval_jitted_one(currSolVar: int, solutionVector: np.array, valuesM: np.array):
    currVal = solutionVector[currSolVar]
    currM = valuesM[currVal]

    return currVal, currM


@njit(cache=True, fastmath=True)
def partial_eval_jitted_two(currIsLegal: bool, currSolVar: int, maxValuesNum: int, binaryConstraintsMatrix: np.array,
                            currVal: int, gradesVector: np.array, solutionVector: np.array,
                            currM: np.array, amount: int, testPrio:int,
                            testvaluesB: np.array, testvaluesQ: np.array, testvaluesP: np.array):
    """Evaluating the critical part of the Evaluation function.
       using numba to speed things up.

    Args:
        currIsLegal (bool): boolean flag to check if the current variable is legal to use.
        currSolVar (int): current place in solutionVector.
        maxValuesNum (int): maximum value number.
        binaryConstraintsMatrix (ndarray): constrains matrix represented as 1d array.
        currVal (int): current value in solution vector.
        gradesVector (List[int]): Grades numpy 1d vector
        solutionVector (List[int]): Solution numpy 1d vector.

    Returns:
        currIsLegal (bool): True if current variable is legal, else False.
    """
    if amount:
        currPrio = testPrio
        currB = testvaluesB[currVal]
        currQ = testvaluesQ[currVal]
        currP = testvaluesP[currVal]

        for pastSolVar in range(currSolVar):
            pastVal = solutionVector[pastSolVar]
            x = currSolVar * maxValuesNum + currVal
            y = pastSolVar * maxValuesNum + pastVal
            currBinaryVal = binaryConstraintsMatrix[x * MAX_TOTAL_VALUES + y]  # faster without function call

            if currBinaryVal == 0:
                currIsLegal = False
                break

            gradesVector[LEVEL_OF_BINARY_CONSTRAINTS] -= currBinaryVal

        if currIsLegal:
            currM -= 1
            gradesVector[2 * currPrio] -= 1
            gradesVector[2 * currPrio + 1] -= currP
            gradesVector[LEVEL_OF_B] -= currB
            gradesVector[LEVEL_OF_Q] -= currQ


@njit(cache=True)
def update_newState_solVec_jitted(action: np.array, newStateSolVec: np.array):
    for idx, val in action:
        newStateSolVec[idx] = val
    return newStateSolVec

#############################################################################################
######################## backup original functions ##########################################
#############################################################################################


# def evaluateSolution(self, solutionVec):
#     """Evaluating the passed SolutionVector as GradesVector.
#
#     Args:
#         solutionVec (SolutionVector): current SolutionVector.
#     Returns:
#         GradesVector: evaluation of passed SolutionVector.
#     """
#     outputEvaluation = GradesVector()
#     MsUsage = deepcopy(self.Ms)
#
#     for currSolVar in range(self.valuesPerVariables.validVarAmount):
#         # currIsLegal = True
#         # currVal = solutionVec.solutionVector[currSolVar]
#         # currM = self.valuesPerVariables.varsData[currSolVar].valuesM[currVal]
#
#         # if MsUsage[currM].amount == 0:
#         #     continue

#         # currPrio = self.valuesPerVariables.varsData[currSolVar].ucPrio
#         # currB = self.valuesPerVariables.varsData[currSolVar].valuesB[currVal]
#         # currQ = self.valuesPerVariables.varsData[currSolVar].valuesQ[currVal]
#         # currP = self.valuesPerVariables.varsData[currSolVar].valuesP[currVal]

#         # for pastSolVar in range(currSolVar):
#         #     pastVal = solutionVec.solutionVector[pastSolVar]
#         #     x = currSolVar * self.maxValuesNum + currVal
#         #     y = pastSolVar * self.maxValuesNum + pastVal
#         #     currBinaryVal = self.binaryConstraintsMatrix[x * MAX_TOTAL_VALUES + y]  # faster without function call
#         #     # currBinaryVal = self.binaryConstraintsMatrix[self.binConsIdx(x, y)]  # with function call
#         #
#         #     if currBinaryVal == 0:
#         #         currIsLegal = False
#         #         break
#         #     outputEvaluation.gradesVector[LEVEL_OF_BINARY_CONSTRAINTS] -= currBinaryVal

#         # if currIsLegal:
#         #     MsUsage[currM].amount -= 1
#         #     outputEvaluation.gradesVector[2 * currPrio] -= 1
#         #     outputEvaluation.gradesVector[2 * currPrio + 1] -= currP
#         #     outputEvaluation.gradesVector[LEVEL_OF_B] -= currB
#         #     outputEvaluation.gradesVector[LEVEL_OF_Q] -= currQ
#
#     return outputEvaluation


# def actions(self, state):
#     """Generates self.availableStatesSize random 2-tuples.
#     what index to change and into what number. because it is random it doesn't matter what is the current state.
#
#
#     Args:
#         state (SolutionVector): current state.
#
#     Returns:
#         list[list[(int, int)]]: a single action is performing self.numOfVarChangesInNeighborhood in current state.
#             and there are self.availableStatesSize such actions.
#     """
#     actions = actions_jitted(self.availableStatesSize,
#                              self.valuesPerVariables.validVarAmount,
#                              self.numOfVarChangesInNeighborhood,
#                              self.valuesPerVariables.extract_valuesAmount())
#
#     # start = time.time()
#     # actions = []
#     # for action in range(self.availableStatesSize):  # change (rand_idx, rand_val)
#     #     change = []
#     #     for var in range(1, self.numOfVarChangesInNeighborhood + 1):
#     #         rand_entry = np.random.randint(0, self.valuesPerVariables.validVarAmount)
#     #         rand_val = np.random.randint(0, self.valuesPerVariables.varsData[rand_entry].valuesAmount)
#     #         change.append((rand_entry, rand_val))
#     #     actions.append(change)
#     # print(time.time() - start)
#
#     return actions

# def result(self, state, action):
#     """Applying action to the current state. meaning applying delta changed which are located in action.
#
#     Args:
#         state (SolutionVector): current SolutionVector.
#         action (list[(int, int)]): what index to perform the change on, the change.
#
#     Returns:
#         SolutionVector: new state after performing an action on it.
#     """
#     newState = deepcopy(state)
#     newState.solutionVector = update_newState_solVec_jitted(action, newState.solutionVector)
#     return newState
#
#     # for idx, val in action:
#     #     newState.solutionVector[idx] = val
#     # return newState
#
# def value(self, state):
#     """Evaluating the current state.
#
#     Args:
#         state (SolutionVector): current state .
#
#     Returns:
#         float: scalarized GradesVector resulted from applying self.evaluateSolution on input state.
#     """
#     evaluation = self.evaluateSolution(state)
#     # works faster with independent function outside the class. also can njit it with cache.
#     scalar = scalarize_jitted(evaluation.gradesVector, evaluation.valuesRange)
#     # return evaluation.scalarize()
#     return scalar