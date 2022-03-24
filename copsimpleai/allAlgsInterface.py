import sys
sys.path.append(r'C:\Users\evgni\Desktop\projects_mine\ref\ref')

from cop import COP
from simpleaipack.search import SearchProblem
from simpleaipack.search.local import beam_best_first, hill_climbing_stochastic, simulated_annealing
from time import time


if __name__ == '__main__':
    path = r'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\CPP_Problems\\'
    alg_name = sys.argv[1]
    problem_seed = sys.argv[2]
    run_time = sys.argv[3]
    algo_seed = sys.argv[4]

    result = None
    problem = None

    if sys.argv[1] == "SHC":
        neighborhood = 0
        args_list = sys.argv
        for idx, arg in enumerate(args_list):
            if arg == '-neighborhood':
                neighborhood = args_list[idx + 1]
        if neighborhood == 0:
            print("-neighborhood is missing!")

        print("Stochastic Hill Climbing")
        print("------------------------")

        problem = COP(problemSeed=int(problem_seed),
                      numOfVarChangesInNeighborhood=int(neighborhood),
                      path=path,
                      algoName=alg_name,
                      algoSeed=algo_seed,
                      initialSolution=None,
                      loadProblemFromFile=True)
        result = hill_climbing_stochastic(problem, iterations_limit=2000, max_run_time=float(run_time), seed=int(algo_seed))

    if sys.argv[1] == "SA":
        neighborhood = 0
        initTemp = None
        tempStep = None
        args_list = sys.argv
        for idx, arg in enumerate(args_list):
            if arg == '-neighborhood':
                neighborhood = args_list[idx + 1]
            if arg == '-inittemp':
                initTemp = args_list[idx + 1]
            if arg == '-tempstep':
                tempStep = args_list[idx + 1]

        if neighborhood == 0:
            print("-neighborhood is missing!")
        if initTemp is None:
            print("-initTemp is missing!")
        if tempStep is None:
            print("-tempStep is missing!")

        print("Simulated Annealing")
        print("-------------------")

        problem = COP(problemSeed=int(problem_seed),
                      numOfVarChangesInNeighborhood=int(neighborhood),
                      path=path,
                      algoName=alg_name,
                      algoSeed=algo_seed,
                      initialSolution=None,
                      loadProblemFromFile=True)
        result = simulated_annealing(problem, iterations_limit=2000, max_run_time=float(run_time),
                                     seed=int(algo_seed), initTemp=int(initTemp), tempStep=int(tempStep))
    if sys.argv[1] == "LBS":
        neighborhood = 0
        args_list = sys.argv
        for idx, arg in enumerate(args_list):
            if arg == '-neighborhood':
                neighborhood = args_list[idx + 1]

        if neighborhood == 0:
            print("-neighborhood is missing!")

        print("Local Beam Search")
        print("-----------------")

        problem = COP(problemSeed=int(problem_seed),
                      numOfVarChangesInNeighborhood=int(neighborhood),
                      path=path,
                      algoName=alg_name,
                      algoSeed=algo_seed,
                      initialSolution=None,
                      loadProblemFromFile=True)
        result = beam_best_first(problem, iterations_limit=2000, max_run_time=float(run_time), seed=int(algo_seed))

    print(f"Results for ParamILS: SAT, -1, {problem_seed}, {result.value}, {algo_seed}")


    # path = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\Problems\\'
    # problem_seed = 500
    # algo_seed = 100
    # time_begining_init = time()
    # copProblem = COP(problemSeed=problem_seed,
    #                  numOfVarChangesInNeighborhood=5,
    #                  path=path,
    #                  initialSolution=None,
    #                  loadProblemFromFile=True)
    #
    # print(f"init time: {time() - time_begining_init}")
    # problem = copProblem
    # time_begining_init = time()
    # # result = hill_climbing_stochastic(problem, iterations_limit=200, max_run_time=20, seed=algo_seed)
    # result = simulated_annealing(problem, iterations_limit=1120, max_run_time=20, seed=algo_seed, initTemp=5,
    #                              tempStep=4)
    # # result = beam_best_first(problem, beam_size=100, iterations_limit=2000, max_run_time=20, seed=algo_seed)
    # print(f"algo run time: {time() - time_begining_init}")
    # print(result)
    # print(type(result))
    # print(result.value)
    # print(result.state)
    # print(problem.evaluateSolution(result.state).scalarize())