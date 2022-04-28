import os
import optuna
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from .allAlgsOptunaFunc import *


optuna_parametrization_funcs = {'SLBS': stochastic_local_beam_search_optuna,
                                'SA': simulated_annehiling_optuna,
                                'SHC': stochastic_hill_climb_optuna,
                                'RW': random_walk_optuna,
                                'RS': random_search_or_greedy_optuna,
                                'GREAT_DELUGE': great_deluge_optuna,
                                'TS': tabu_search_optuna,
                                'CE': cross_entropy_optuna,
                                'GREEDY': random_search_or_greedy_optuna,
                                'LBS': stochastic_local_beam_search_optuna,
                                'GREEDY+GREAT_DELUGE': great_deluge_optuna,
                                'GREEDY+SLBS': stochastic_local_beam_search_optuna,
                                'GREEDY+RS': random_search_or_greedy_optuna,
                                'GREEDY+RW': random_walk_optuna,
                                'GREEDY+SHC': stochastic_hill_climb_optuna,
                                'GREEDYLOOP': random_search_or_greedy_optuna,
                                'GREEDY+TS': tabu_search_optuna,
                                'GREEDY+SA': simulated_annehiling_optuna,
                                'GREEDY+CE': cross_entropy_optuna,
                                'GREEDY+LBS': stochastic_local_beam_search_optuna}


class RepeatPruner(optuna.pruners.BasePruner):
    def prune(self, study, trial):
        trials = study.get_trials(deepcopy=False)
        completed_trials = [t.params for t in trials if t.state == optuna.structs.TrialState.COMPLETE]
        n_trials = len(completed_trials)

        if n_trials == 0:
            return False

        if trial.params in completed_trials:
            return True

        return False


def best_params_for_all_algos(path, output_path, algorithms, problem_seeds, algo_seed, num_iterations, python=False, run_time=1.0,
                              unique_trials=False):
    """Running Optuna parameter optimized for finding the optimal parameters for each algorithm given as argument.
    recording the best result as float of each algorithm at each trial run inside a .txt file
    number of lines in each file is as the #num_iterations.
    finally recording the optimal parameters for each algorithm inside a .txt file, a single .txt file created
    for each problem seed. each .txt file contains all algorithms ran with optimal parameters.
    Added feature, will run "num_iterations" unique trials.

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
        if program crashes for some reason, it will start from where it left off.

    """
    def print_best_callback(study, trial, path, algo, prob_seed):
        """Appending to a file the last study.best_value in order to create an increasing monotone graph for
        Optuna performance for for finding the best possible parameters for the problem.
        Added feature, will store only the best "num_iterations" lines in file.

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

        with open(path + rf'best_values_for_algs/bestValue_for_{algo}_problem_{prob_seed}.txt', "a+") as txt_file:
                txt_file.write(str(study.best_value) + '\n')
                txt_file.seek(0)
                data = txt_file.readlines()
                n_lines = len(data)
                if n_lines > num_iterations:
                    txt_file.seek(0)
                    txt_file.truncate()
                    txt_file.writelines(data[1:])

    direction = 'maximize' if python else 'minimize'
    n_jobs = 6

    if not python:
        files_path = os.listdir(os.path.join(os.getcwd(), '..', 'copsimpleai', 'LocalSearch', f'algo_seed_{algo_seed}', 'BestParamsPerAlgo'))
        file_names = [file for file in files_path if '.txt' in file and 'run' not in file]
    else:
        files_path = os.listdir(os.path.join(os.getcwd(), '..', 'copsimpleai', 'pythonLocalSearch', f'algo_seed_{algo_seed}', 'BestParamsPerAlgo'))
        file_names = [file for file in files_path if '.txt' in file and 'run' not in file]

    available_problems_seeds = [file.split('_')[-1][:-4] for file in file_names]  # recover problem seed from file name
    missing_problem_seeds = list(set(problem_seeds) - set(available_problems_seeds))

    for problem_seed in missing_problem_seeds:
        temp_arr = []
        best_params_per_algo_temp = []
        for algo in algorithms:
            run_optuna_func = partial(optuna_parametrization_funcs[algo], path=path, algo=algo, problem_seed=problem_seed, algo_seed=algo_seed, python=python, run_time=run_time)
            call_back_func = partial(print_best_callback, path=output_path, algo=algo, prob_seed=problem_seed)
            study = optuna.create_study(study_name='test', direction=direction, pruner=RepeatPruner())
            if unique_trials:
                iter = 0
                while num_iterations > len(set(str(t.params) for t in study.trials)):
                    study.optimize(run_optuna_func, n_trials=n_jobs, callbacks=[call_back_func], n_jobs=n_jobs)  # -1 means maximum cpu capacity
                    iter += n_jobs
                    if algo in ['GREEDY', 'RS', 'GREEDYLOOP', 'GREEDY+RS'] and iter >= num_iterations:
                        break
            else:
                study.optimize(run_optuna_func, n_trials=num_iterations, callbacks=[call_back_func], n_jobs=n_jobs)
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


def run_algs_with_optimal_params(run_file, path, python, num_workers, algo_seed):
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

    Notes:
        if program crashes for some reason, it will start from where it left off.
    """

    if not python:
        file_names = os.listdir(os.path.join(os.getcwd(), '..', 'copsimpleai', 'LocalSearch', 'Results'))
    else:
        file_names = os.listdir(os.path.join(os.getcwd(), '..', 'copsimpleai', 'pythonLocalSearch', 'Results'))

    available_files = [(file.split('_')[1], file.split('_')[3]) for file in file_names if 'BestValue' in file]
    ran_files_with_optimal_params = []
    files_to_run = []
    with open(run_file, "r") as f:
        for line in f:
            files_to_run.append(line)
            for file in available_files:
                if file[0] + ',' + file[1] in line:
                    ran_files_with_optimal_params.append(file)
                    break

    missing_files_to_run = []
    for alg in files_to_run:
        for existing_alg in ran_files_with_optimal_params:
            if existing_alg[0] + ',' + existing_alg[1] in alg:
                missing_files_to_run.append(alg)
                break

    missing_files_to_run = list(set(files_to_run) - set(missing_files_to_run))
    run_que = Queue()
    for params_line in missing_files_to_run:
        run_que.put(params_line.split(','))

    # with open(run_file, "r") as f:
    #     for line in f:
    #         run_que.put(line.split(','))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(num_workers):
            executor.submit(best_params_worker, path, run_que, python)  #add algo seed path


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

    Notes:
        if program crashes for some reason, it will start from where it left off.
    """
    file_names = os.listdir(os.path.join(os.getcwd(), '..', 'copsimpleai', 'CPP_Problems'))
    available_problems = []
    run_que = Queue()
    for problem in problem_seeds:
        for file in file_names:
            if problem in file:
                available_problems.append(problem)
                break

    missing_problems = list(set(problem_seeds) - set(available_problems))

    for problem in missing_problems:
        run_que.put(("dummy", str(problem)))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _ in range(num_workers):
                executor.submit(problemCreatorWorker, path, run_que)
