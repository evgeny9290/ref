"""
file containing all constants required for the COP problem.
"""

MAX_VAR_PRIORITIES = 20  # must link to VALUES_RANGE[] array somehow. what is variable priority? Condition Variables prio?
MAX_TOTAL_VALUES = 6000  # upper ceiling of the problem (for matrix of constrains)
MAX_NUM_OF_VARS = 200  # maximum number of Condition Variables
MAX_VALUES_OF_VAR = 200  # maximum domain for every Condition Variable
MAX_NUM_OF_MS = 20  # number of types of resources
MAX_NUM_OF_R_PER_M = 30  # max number of resourse per action/tactic for each type
MAX_LENGTH_OF_GRADES_VECTOR = 19  # number of objectives in object func.
LEVEL_OF_Q = 4  # constrain priority lvl 4. the lower the better
LEVEL_OF_BINARY_CONSTRAINTS = 3  # constrain priority lvl 3
LEVEL_OF_B = 2  # constrain priority lvl 2
PRIORITIES_NUM = 1  # constrain priority lvl 1
MAX_CONSTRAINTS_RATIO = 10  #
NUM_OF_P_VALUES = 10  # image domain is of size 10
NUM_OF_Q_VALUES = 10  # image domain is of size 10
MAX_NUM_OF_ELITE = 10  # Elite size for genetic algorithm