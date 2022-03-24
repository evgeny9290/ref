import subprocess
import optuna
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
# import os


def best_params_for_all_algos(path, output_path, algorithms, problem_seeds, algo_seed, num_iterations, python=False, run_time='1.0'):

    def run_optuna(trial, algo, problem_seed, algo_seed, python, run_time):
        algorithm = algo
        algorithm_seed = algo_seed
        problem_seed = problem_seed
        run_time = run_time
        neighborhood = trial.suggest_int('neighborhood', 10, 30)
        numelites = trial.suggest_int('numelites', 10, 20)
        inittemp = trial.suggest_int('inittemp', 1, 10)
        tempstep = trial.suggest_int('tempstep', 1, 10)
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


# def run_algs_with_optimal_params(run_file, path):
#     run_array = []
#     with open(run_file, "r") as f:
#         for line in f:
#             run_array.append(line.split(','))
#
#     for idx, params in enumerate(run_array):
#         subprocess.run([path, *params])

# def remove_all_files_in_path(path):
#     for file in os.listdir(path):
#         os.remove(path + file)

def best_params_worker(path, run_que, python):
    while not run_que.empty():
        params = run_que.get()
        if python:
            subprocess.run(['python', path, *params])
        else:
            subprocess.run([path, *params])


def run_algs_with_optimal_params(run_file, path, python):
    run_que = Queue()
    with open(run_file, "r") as f:
        for line in f:
            run_que.put(line.split(','))

    with ThreadPoolExecutor(max_workers=5) as executor:
        for _ in range(5):
            executor.submit(best_params_worker, path, run_que, python)


def problemCreatorWorker(path, run_que):
    while not run_que.empty():
        params = run_que.get()
        subprocess.run([path, *params])


def problemCreatorFromCPP(problem_seeds, path):
    run_que = Queue()
    for problem in problem_seeds:
        run_que.put(("dummy", str(problem)))

    with ThreadPoolExecutor(max_workers=5) as executor:
        for _ in range(5):
            executor.submit(problemCreatorWorker, path, run_que)


# if __name__ == '__main__':
    # problemCreatorPath = r'C:\Users\evgni\Desktop\Projects\LocalSearch\Debug\LocalSearchProblemGenerator.exe'
    # problem_seeds = ['182', '271', '291', '375', '390', '504', '549', '567', '643', '805', '1101', '1125', '2923',
    #                  '3562']
    # problemCreatorFromCPP(problem_seeds, problemCreatorPath)

    # run_file = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\BestParamsPerAlgo\final_run.txt'
    # path = r'C:\Users\evgni\Desktop\Projects\LocalSearch\Debug\LocalSearch.exe'
    # output_path = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\BestParamsPerAlgo\\'
    # for_python_final_run_create_problem_set = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\BestParamsPerAlgo\final_run_problem_set.txt'
    # python_output_path = r'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\BestParamsPerAlgo\\'
    # python_path =  r'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\allAlgsInterface.py'
    # results_path = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\Results\\'
    #
    # algorithms = ['GREAT_DELUGE', 'SLBS', 'RS', 'RW', 'SHC', 'GREEDY', 'TS', 'SA', 'CE',
    #               'GREEDY+GREAT_DELUGE', 'GREEDY+SLBS', 'GREEDY+RS', 'GREEDY+RW', 'GREEDY+SHC', 'GREEDYLOOP',
    #               'GREEDY+TS', 'GREEDY+SA', 'GREEDY+CE']  # 1
    #
    # problem_seeds = ['182', '271', '291', '375', '390', '504', '549', '567', '643', '805', '1101', '1125', '2923',
    #                  '3562']
    #
    # algo_seed = '331991908'  # 4
    # num_iterations = 20
    # # best_params_for_all_algos(path, output_path, algorithms, problem_seeds, algo_seed, num_iterations)
    # # run_algs_with_optimal_params(final_run_create_problem_set, path)
    # python_algs = ['SA', 'SHC', 'LBS']
    # python_problem_seeds = ['271', '291', '375', '390', '500', '504', '549', '567', '643', '805',
    #                         '1101', '1125', '2923', '3562']
    # python_algo_seed = '331991908'
    # python_num_iterations = 8
    # best_params_for_all_algos(python_path, python_output_path, python_algs,
    #                           python_problem_seeds, python_algo_seed, python_num_iterations,
    #                           python=True)
    #
    #
    # run_algs_with_optimal_params(for_python_final_run_create_problem_set, path, python=True)