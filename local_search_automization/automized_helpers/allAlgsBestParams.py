import subprocess
import optuna
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from queue import Queue


def best_params_for_all_algos(path, output_path, algorithms, problem_seeds, algo_seed, num_iterations, python=False, run_time=1.0):
    """Running Optuna parameter optimized for finding the optimal parameters for each algorithm given as argument.
    recording the best result as float of each algorithm at each trial run inside a .txt file
    number of lines in each file is as the #num_iterations.
    finally recording the optimal parameters for each algorithm inside a .txt file, a single .txt file created
    for each problem seed. each .txt file contains all algorithms ran with optimal parameters.

    Args:
        path (str): path to the .exe file that needs to be ran for Optuna.
        output_path (str): path to the folder which will store the .txt files with optimal parameters
            for each problem seed for each algorithm.
        algorithms (list[str]): list of algorithm names that are required to run.
        problem_seeds (list[str]): seed numbers representing the given problems.
        algo_seed (str): seed number representing the stochastic behaviour of an algorithm.
        num_iterations (int): number of trials to run for Optuna parameter optimized.
        python (bool, optional): True if running python algorithms from SimpleAi
            False if running CPP algorithms from LocalSearch.
        run_time (float, optional): run time of algorithms.

    Return:
        None.

    Notes:
        see n_jobs parameter inside study.optimize in order to vary number processes ran simultaneously.

    """
    def run_optuna(trial, algo, problem_seed, algo_seed, python, run_time):
        """Executing a single run of an algorithm for some given problem seed.
        this run is essentially a trial run in order to find the future optimal parameters.

        Args:
            trial (int): trial number executed.
            algo (str): algorithm name executed.
            problem_seed (int): given problem seed.
            algo_seed (int): stochastic behavior of an algorithm.
            python (bool): True if running python algorithms from SimpleAi
                False if running CPP algorithms from LocalSearch.
            run_time (float): given run time for algorithm.

        Returns:
             float: Quality of the algorithm for the given trial (scalarization value).
        """
        algorithm = algo
        algorithm_seed = algo_seed
        problem_seed = problem_seed
        run_time = run_time
        neighborhood = trial.suggest_int('neighborhood', 10, 30)
        numelites = trial.suggest_int('numelites', 10, 20)
        inittemp = trial.suggest_float('inittemp', 10, 40)
        tempstep = trial.suggest_float('tempstep', 1, 10)
        tabusize = trial.suggest_int('tabusize', 5, 10)
        samples = trial.suggest_int('samples', 5, 20)
        initsolweight = trial.suggest_float('initsolweight', 0.0, 1.0)
        alpha = trial.suggest_float('alpha', 0.3, 0.9)
        rho = trial.suggest_float('rho', 0.5, 1.0)  # elite size
        epsilon = trial.suggest_float('epsilon', 0.2, 1.0)

        if python:
            x = subprocess.run(['python', path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                                   , '-neighborhood', str(neighborhood), '-inittemp', str(inittemp), '-tempstep'
                                   , str(tempstep), '-numelites', str(numelites), '-tabusize', str(tabusize)
                                   , '-samples', str(samples), '-initsolweight', str(initsolweight), '-alpha', str(alpha)
                                   , '-rho', str(rho), '-epsilon', str(epsilon)]
                                   , capture_output=True, text=True)
        else:
            x = subprocess.run([path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                                   , '-neighborhood', str(neighborhood), '-inittemp', str(inittemp), '-tempstep'
                                   , str(tempstep), '-numelites', str(numelites), '-tabusize', str(tabusize)
                                   , '-samples', str(samples), '-initsolweight', str(initsolweight), '-alpha', str(alpha)
                                   , '-rho', str(rho), '-epsilon', str(epsilon)]
                                   , capture_output=True, text=True)

        return float(str(x).split(',')[-3])  # magic to get the scalarization result

    def print_best_callback(study, trial, path, algo, prob_seed):
        """Appending to a file the last study.best_value in order to create an increasing monotone graph for
        Optuna performance for for finding the best possible parameters for the problem.

        Args:
            study (class): running study variable for extracting the best_value so far.
            trial (int): trial number executed.
            path (str): path to the folder which will store the .txt files with best_value
                for each problem seed for each algorithm.
            algo (str): algorithm name executed.
            prob_seed (int): given problem seed.

        Returns:
             None.
        """
        with open(path + rf'best_values_for_algs\bestValue_for_{algo}_problem_{prob_seed}.txt', "a") as txt_file:
                txt_file.write(str(study.best_value) + '\n')

    direction = 'maximize' if python else 'minimize'

    for problem_seed in problem_seeds:
        temp_arr = []
        best_params_per_algo_temp = []
        for algo in algorithms:
            run_optuna_func = partial(run_optuna, algo=algo, problem_seed=problem_seed, algo_seed=algo_seed, python=python, run_time=run_time)
            call_back_func = partial(print_best_callback, path=output_path, algo=algo, prob_seed=problem_seed)
            study = optuna.create_study(study_name='test', direction=direction)
            study.optimize(run_optuna_func, n_trials=num_iterations, callbacks=[call_back_func], n_jobs=-1)  # -1 means maximum cpu capacity
            best_params_per_algo_temp.append((algo, problem_seed, algo_seed, study.best_trial.params))
        temp_arr.extend(best_params_per_algo_temp)
        with open(output_path + rf'bestParams_problem_{problem_seed}.txt', "w") as txt_file:
            for algo, prob_seed, algo_seed, best_params in temp_arr:
                txt_file.write(str(algo) + "," + str(best_params) + '\n')


def best_params_worker(path, run_que, python):
    """While the run_que is not empty creates a subprocess which solves COP problem running some algorithm
    using .exe found in path.

    Args:
        path (str):  path to the .exe file which is responsible for solving the COP problem running some algorithm.
        run_que (queue[*str]): parameters for the algorithm to be ran using the .exe in path.
        python (bool): True if running python algorithms from SimpleAi, False if running CPP algorithms from LocalSearch.

    Returns:
        None.
    """
    while not run_que.empty():
        params = run_que.get()
        if python:
            subprocess.run(['python', path, *params])
        else:
            subprocess.run([path, *params])


def run_algs_with_optimal_params(run_file, path, python, num_workers):
    """Fills a Queue named "run_que" with a tuple (*params) according to the parameters presented in each line
    inside the .txt file located in path.
    when run_que is filled calls "best_params_worker" function with path, run_que, python as params.

    Args:
        run_file (str):  path to the .txt file where each line contains the OPTIMAL parameters req for the algorithm.
        path (str): path to the .exe file which is responsible for running the algorithm
          using the params in specific line inside run_file.
        python (bool): True if running python algorithms from SimpleAi, False if running CPP algorithms from LocalSearch.
        num_workers (int): number of threads used for running algorithms simultaneously.

    Returns:
        None.
    """
    run_que = Queue()
    with open(run_file, "r") as f:
        for line in f:
            run_que.put(line.split(','))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(num_workers):
            executor.submit(best_params_worker, path, run_que, python)


def problemCreatorWorker(path, run_que):
    """While the run_que is not empty creates a subprocess which creates a COP problem using .exe found in path.

    Args:
        path (str):  path to the .exe file which is responsible for problem creating using some seed.
        run_que (queue[(str, str)]): parameters for the problem to be created using the .exe in path.

    Returns:
        None.
    """
    while not run_que.empty():
        params = run_que.get()
        subprocess.run([path, *params])


def problemCreatorFromCPP(problem_seeds, path, num_workers):
    """Fills a Queue named "run_que" with a 2-tuple (str, str(problem_seed)) as the .exe file takes 2 parameters.
    when run_que is filled calls "problemCreatorWorker" function with path, run_que as params.

    Args:
        problem_seeds (list[str]): seed numbers representing the random problem created.
        path (str): path to the .exe file which is responsible for problem creating using some seed.
        num_workers (int): number of threads used for creating problems simultaneously.

    Returns:
        None.
    """
    run_que = Queue()
    for problem in problem_seeds:
        run_que.put(("dummy", str(problem)))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(num_workers):
            executor.submit(problemCreatorWorker, path, run_que)
