import math
import subprocess
import optuna
from functools import partial
import numpy as np
import random
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor

from automized_helpers.allAlgsBestParams import run_algs_with_optimal_params
from automized_helpers.bestParams_final_run_file import automize_final_run_file
from automized_helpers.create_expected_optuna_graph import automize_optuna_expected_graph
from automized_helpers.create_graphs_and_expected_graph import automize_expected_graph
from automized_helpers.allAlgsBestParams import best_params_for_all_algos

###################################################################################################
###################################################################################################
#####################################  READ THIS  #################################################
# for now this can be run everytime with your algs/problem_set_seeds/problem_seed everytime
# iff you clear all the paths req.
# can easily be modified using the remove_all_files_in_path in the class in needed placed so no need to delete manually

# what is in need of clearing are the following folders:
# "Results", "BestParamsPerAlgo", "BestParamsPerAlgo\best_values_for_algs"

# only txt files need clearing, not any folder inside!
###################################################################################################
###################################################################################################
###################################################################################################


class COPLocalSearchAlgorithmsAveraging:
    def __init__(self, results_path, run_file_path, LS_exe, best_params_path, best_val_for_alg_path, algorithms, problem_seeds, algo_seed, num_iterations):
        self.results_path = results_path  # \\results\\
        self.run_file_path = run_file_path  # \\final_run.txt
        self.LS_exe = LS_exe  # \\LocalSearch.exe
        self.best_params_path = best_params_path  # \\BestParamsPerAlgo\\'
        self.algorithms = algorithms  # list of algorithms
        self.problem_seeds = problem_seeds  # list of problem seeds
        self.algo_seed = algo_seed  # single seed (can be easily modified later)
        self.best_val_for_alg_path = best_val_for_alg_path  # \\BestParamsPerAlgo\best_values_for_algs\\
        self.num_algos = len(algorithms)
        self.num_problems = len(problem_seeds)
        self.num_iterations = num_iterations

    def create_final_run_and_run_algs_from_final_run(self, ran_optimal_params):  # both create final_run file if doesnt exist and run prog with those params
        if not os.path.exists(self.run_file_path):
            automize_final_run_file(self.best_params_path, self.problem_seeds)
        if not ran_optimal_params:
            run_algs_with_optimal_params(self.run_file_path, self.LS_exe)

    def _run_with_best_params_and_create_expected_graphs(self, print_graphs_bool, ran_optimal_params):  # private func for find_best_params_run_them_output_expected_graphs
        self.create_final_run_and_run_algs_from_final_run(ran_optimal_params)
        self.create_expected_graphs(print_graphs_bool)
        # automize_optuna_expected_graph(self.best_val_for_alg_path, self.num_algos, self.num_problems, self.num_iterations)
        # automize_expected_graph(self.results_path, self.num_algos, self.num_problems, self.problem_seeds, print_graphs_bool)

    @staticmethod
    def remove_all_files_in_path(path):
        for file in os.listdir(path):
            os.remove(path + file)

    def find_best_params_run_then_output_expected_graphs(self, print_graphs_bool, ran_optimal_params):  # both create final_run + run with best params + create graphs
        if not ran_optimal_params:
            self.remove_all_files_in_path(self.results_path)  # clear results folder
        self._run_with_best_params_and_create_expected_graphs(print_graphs_bool, ran_optimal_params)  # run prog with best params and create expected graphs

    def run_optuna_param_optimization(self):  # only running optuna
        best_params_for_all_algos(self.LS_exe, self.best_params_path, self.algorithms,
                                  self.problem_seeds, self.algo_seed, self.num_iterations)

    def create_expected_graphs(self, print_graphs_bool):  # only expected_graphs and graphs for each problem if needed
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(automize_optuna_expected_graph, self.best_val_for_alg_path, self.num_algos, self.num_problems, self.num_iterations)
            executor.submit(automize_expected_graph, self.results_path, self.num_algos, self.num_problems, self.problem_seeds, print_graphs_bool)
        # automize_optuna_expected_graph(self.best_val_for_alg_path, self.num_algos, self.num_problems, self.num_iterations)
        # automize_expected_graph(self.results_path, self.num_algos, self.num_problems, self.problem_seeds, print_graphs_bool)

    def create_final_run_file(self):  # only final_run file
        automize_final_run_file(self.best_params_path, self.problem_seeds)


if __name__ == '__main__':
    results_path = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\Results\\'
    run_file = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\BestParamsPerAlgo\final_run.txt'
    LS_exe = r'C:\Users\evgni\Desktop\Projects\LocalSearch\Debug\LocalSearch.exe'
    path_best_args = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\BestParamsPerAlgo\best_values_for_algs\\'
    best_params_path = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\BestParamsPerAlgo\\'

    num_iterations = 24  # used for optuna as number of trials, has to be accurate
    algo_seed = '331991908'
    problem_set = ['182', '271', '291', '375', '390', '504', '549', '567', '643', '805', '1101', '1125', '2923', '3562']
    algorithms = ['GREAT_DELUGE', 'SLBS', 'RS', 'RW', 'SHC', 'GREEDY', 'TS', 'SA', 'CE',
                  'GREEDY+GREAT_DELUGE', 'GREEDY+SLBS', 'GREEDY+RS', 'GREEDY+RW', 'GREEDY+SHC', 'GREEDYLOOP',
                  'GREEDY+TS', 'GREEDY+SA', 'GREEDY+CE']

    # algo_seed = '331991908' # 4
    # num_iterations = 5
    # algorithms = ['GREAT_DELUGE', 'SLBS', 'GREEDY+GREAT_DELUGE', 'GREEDY+SLBS']  # 1
    # problem_set = ['504', '2923']

    COP_automized_run = COPLocalSearchAlgorithmsAveraging(results_path=results_path,
                                                          run_file_path=run_file,
                                                          LS_exe=LS_exe,
                                                          best_params_path=best_params_path,
                                                          best_val_for_alg_path=path_best_args,
                                                          algorithms=algorithms,
                                                          problem_seeds=problem_set,
                                                          algo_seed=algo_seed,
                                                          num_iterations=num_iterations)

    # COP_automized_run.run_optuna_param_optimization()  # run this if first you want to know the optimal params
    COP_automized_run.find_best_params_run_then_output_expected_graphs(print_graphs_bool=False, ran_optimal_params=True)  # if optimal params already exist run this
    # COP_automized_run.create_expected_graphs(print_graphs_bool=False)