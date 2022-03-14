import math
import subprocess
import optuna
from functools import partial
import numpy as np
import random
import pandas as pd
import os


def best_params_for_all_algos(path, output_path, algorithms, problem_seeds, algo_seed, num_iterations):

    def run_optuna(trial, algo, problem_seed, algo_seed):
        algorithm = algo
        algorithm_seed = algo_seed
        problem_seed = problem_seed
        run_time = '1.0'
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

        x = subprocess.run([path, algorithm, problem_seed, run_time, algorithm_seed
                               , '-neighborhood', str(neighborhood), '-inittemp', str(inittemp), '-tempstep'
                               , str(tempstep), '-numelites', str(numelites), '-tabusize', str(tabusize)
                               , '-samples', str(samples), '-initsolweight', str(initsolweight), '-alpha', str(alpha)
                               , '-rho', str(rho), '-epsilon', str(epsilon)]
                               , capture_output=True, text=True)
        return float(str(x).split(',')[-3])  # magic to get the scalarization result

    def print_best_callback(study, trial, path, algo, prob_seed):
        with open(path + rf'best_values_for_algs\bestValue_for_{algo}_problem_{prob_seed}.txt', "a") as txt_file:
                txt_file.write(str(study.best_value) + '\n')

    for problem_seed in problem_seeds:
        temp_arr = []
        best_params_per_algo_temp = []
        for algo in algorithms:
            run_optuna_func = partial(run_optuna, algo=algo, problem_seed=problem_seed, algo_seed=algo_seed)
            call_back_func = partial(print_best_callback, path=output_path, algo=algo, prob_seed=problem_seed)
            study = optuna.create_study(direction='minimize')
            study.optimize(run_optuna_func, n_trials=num_iterations, callbacks=[call_back_func])
            best_params_per_algo_temp.append((algo, problem_seed, algo_seed, study.best_trial.params))
        temp_arr.extend(best_params_per_algo_temp)
        with open(output_path + rf'bestParams_problem_{problem_seed}.txt', "w") as txt_file:
            for algo, prob_seed, algo_seed, best_params in temp_arr:
                txt_file.write(str(algo) + "," + str(best_params) + '\n')


def run_algs_with_optimal_params(run_file, path):
    run_array = []
    with open(run_file, "r") as f:
        for line in f:
            run_array.append(line.split(','))

    for idx, params in enumerate(run_array):
        subprocess.run([path, *params])


if __name__ == '__main__':
    run_file = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\BestParamsPerAlgo\final_run.txt'
    path = r'C:\Users\evgni\Desktop\Projects\LocalSearch\Debug\LocalSearch.exe'
    output_path = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\BestParamsPerAlgo\\'

    algorithms = ['GREAT_DELUGE', 'SLBS', 'RS', 'RW', 'SHC', 'GREEDY', 'TS', 'SA', 'CE',
                  'GREEDY+GREAT_DELUGE', 'GREEDY+SLBS', 'GREEDY+RS', 'GREEDY+RW', 'GREEDY+SHC', 'GREEDYLOOP',
                  'GREEDY+TS', 'GREEDY+SA', 'GREEDY+CE']  # 1
    problem_seeds = ['182', '271', '291', '375', '390', '504', '549', '567', '643', '805', '1101', '1125', '2923',
                     '3562']
    algo_seed = '331991908' # 4
    num_iterations = 20

    best_params_for_all_algos(path, output_path, algorithms, problem_seeds, algo_seed, num_iterations)

    # run_algs_with_optimal_params(run_file, path)