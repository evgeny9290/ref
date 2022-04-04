import os
import random

from automized_helpers.allAlgsBestParams import run_algs_with_optimal_params
from automized_helpers.bestParams_final_run_file import automize_final_run_file
from automized_helpers.create_expected_optuna_graph import automize_optuna_expected_graph
from automized_helpers.create_graphs_and_expected_graph import automize_expected_graph
from automized_helpers.allAlgsBestParams import best_params_for_all_algos
from automized_helpers.allAlgsBestParams import problemCreatorFromCPP
from automized_helpers.combine_expected_graphs import automize_combined_expected_graphs

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
    def __init__(self, problemGenerator_exe_path, results_path, run_file_path,
                 LS_exe, best_params_path, best_val_for_alg_path, algorithms,
                 problem_seeds, algo_seed, num_iterations, run_time, graphs_path,
                 cpp_dataframes_path=".", python_dataframes_path=".",
                 python=False, num_workers=1, backup=False):
        """This class is responsible for automizing the whole process.
        Be it running Optuna in order to find the optimal parameters for each algorithm, executing algorithms
        with its optimal parameters, creating expected graphs for Optuna performance and algorithms from those optimal runs.

        Args:
            problemGenerator_exe_path (str): path to the CPP .exe file responsible for COP problem creation.
            results_path (str): path to the folder which contains the results of every algorithm.
            run_file_path (str): path to the .txt file where each line contains the OPTIMAL parameters req for the algorithm.
            LS_exe (str): path to the .exe file that needs to be ran for Optuna.
                .py file script if want to run python algorithms.
            best_params_path (str): path to the folder which has all file problems with best parameters
                for each problem seed for each algorithm
            best_val_for_alg_path (str): path to folder containing files with best_values so far from Optuna.
            algorithms (list[str]): array of algorithm names.
            problem_seeds (list[str]): array of problem seeds.
            algo_seed (str): seed for stochastic behaviour of algorithm.
            num_iterations (int): number of iterations which Optuna ran for.
            run_time (float): run time parameter for every algorithm.
            graphs_path (str): path to the folder which will contain the expected graph.
            cpp_dataframes_path (str, optional): path to where the backup DataFrames will saved to if comes from cpp.
            python_dataframes_path (str, optional): path to where the backup DataFrames will saved to if comes from python.
            python (bool, optional):  True if running python algorithms from SimpleAi
                False if running CPP algorithms from LocalSearch.
            num_workers (int, optional): number of thread workers responsible for running CPP problemCreation
                and also number of thread workers responsible for executing algorithms with their optimal parameters.
            backup (bool, optional): True if want to same all DataFrames as .csv in appropriate path else False.

        Returns:
            None.
        """
        self.backup = backup
        self.problemGenerator_exe_path = problemGenerator_exe_path
        self.python = python
        self.run_time = run_time
        self.cpp_dataframes_path = cpp_dataframes_path
        self.python_dataframes_path = python_dataframes_path
        self.graphs_path = graphs_path  # \\graphs\\
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
        self.num_workers = num_workers

    def create_final_run_and_run_algs_from_final_run(self, ran_optimal_params):  # both create final_run file if doesnt exist and run prog with those params
        """If file doesnt exist Creates "final run" .txt file where each line contains the optimal parameters found for a specific algorithm via Optuna.
         If didn't run commands from "final run" .txt file first execute them.

        Args:
            ran_optimal_params (bool): True if already executed algorithms with optimal parameters, else False.

        Returns:
             None.
        """
        if not os.path.exists(self.run_file_path):
            automize_final_run_file(self.best_params_path, self.problem_seeds, self.algo_seed, self.python, self.run_time)
        if not ran_optimal_params:
            run_algs_with_optimal_params(self.run_file_path, self.LS_exe, self.python, self.num_workers)

    def _run_with_best_params_and_create_expected_graphs(self, print_graphs_bool, ran_optimal_params):  # private func for find_best_params_run_them_output_expected_graphs
        """Automize whole process.
         Creating "final run" .txt file, running algorithms with optimal parameters,
         creating expected optuna graph, creating expected graph.

        Args:
            print_graphs_bool (bool): True if want to print all graphs for every problem else False.
            ran_optimal_params (bool): True if already executed algorithms with optimal parameters, else False.

        Returns:
            None.
        """
        self.create_final_run_and_run_algs_from_final_run(ran_optimal_params)
        self.create_expected_graphs(print_graphs_bool)

    @staticmethod
    def remove_all_files_in_path(path):
        """Static method. Removes all files in path directory.

        Args:
            path (str): path to the directory where you want your files removed from.

        Returns:
            None.
        """
        for file in os.listdir(path):
            os.remove(path + file)

    def find_best_params_run_then_output_expected_graphs(self, print_graphs_bool, ran_optimal_params):  # both create final_run + run with best params + create graphs
        """First remove all unwanted files from results folder if didn't run algorithms with optimal parameters.
        Then call function to automize whole process of creating run_file, running it, creating both expected graphs for optuna and algorithms.

        Args:
            print_graphs_bool (bool): True if want to print all graphs for every problem else False.
            ran_optimal_params (bool): True if already executed algorithms with optimal parameters, else False.

        Returns:
             None.
        """
        if not ran_optimal_params:
            self.remove_all_files_in_path(self.results_path)  # clear results folder
        self._run_with_best_params_and_create_expected_graphs(print_graphs_bool, ran_optimal_params)  # run prog with best params and create expected graphs

    def run_optuna_param_optimization(self):  # only running optuna
        """Run optuna parameter optimized for algorithms passed at class creation.

        Args:
            None.

        Returns:
             None.
        """
        best_params_for_all_algos(self.LS_exe, self.best_params_path, self.algorithms,
                                  self.problem_seeds, self.algo_seed, self.num_iterations, self.python, self.run_time)

    def create_expected_graphs(self, print_graphs_bool):  # only expected_graphs and graphs for each problem if needed
        """Calls upon functions responsible for Creating expected graphs, both for Optuna and all Algorithms.

        Args:
            print_graphs_bool (bool): True if want to print all graphs for every problem else False.

        Returns:
            None.
        """
        automize_optuna_expected_graph(self.best_val_for_alg_path, self.num_algos, self.num_problems,
                                       self.num_iterations, self.graphs_path,  self.python)
        automize_expected_graph(self.results_path, self.num_algos, self.num_problems,
                                self.problem_seeds, print_graphs_bool, self.graphs_path, self.python, self.run_time,
                                self.cpp_dataframes_path, self.python_dataframes_path, self.backup)

    def create_final_run_file(self):  # only final_run file
        """Creates "final run" .txt file where each line contains the optimal parameters found for a specific algorithm via Optuna.

        Args:
            None.

        Returns:
            None.
        """
        automize_final_run_file(self.best_params_path, self.problem_seeds, self.algo_seed, self.python, self.run_time)

    def generate_problems_from_seeds(self):
        """Creating COP problems from cpp .exe file where the path to it is passed on class creation.

        Args:
            None.

        Returns:
            None.
        """
        problemCreatorFromCPP(self.problem_seeds, self.problemGenerator_exe_path, self.num_workers)

    def combined_expected_graphs(self):
        """Creating combined expected graph for both cpp,python algorithms for the exact same problems.

        Args:
            None.

        Returns:
            None.
        """
        automize_combined_expected_graphs(self.cpp_dataframes_path + "expected_dataframes/",
                                          self.python_dataframes_path + "expected_dataframes/",
                                          self.graphs_path)


if __name__ == '__main__':
    # problemCreatorPath = r'C:\Users\evgni\Desktop\Projects\LocalSearch\Debug\LocalSearchProblemGenerator.exe'
    # graphs_path = r'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\graphs\\'
    # cpp_dataframes_path = r'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\cpp_dataframes\\'
    # results_path = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\Results\\'
    # run_file = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\BestParamsPerAlgo\final_run.txt'
    # LS_exe = r'C:\Users\evgni\Desktop\Projects\LocalSearch\Debug\LocalSearch.exe'
    # path_best_args = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\BestParamsPerAlgo\best_values_for_algs\\'
    # best_params_path = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\BestParamsPerAlgo\\'
    # python_dataframes_path = r'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\python_dataframes\\'

    problemCreatorPath = r'../LocalSearchProblemGenerator/Debug/LocalSearchProblemGenerator.exe'
    graphs_path = r'../copsimpleai/graphs/'
    cpp_dataframes_path = r'../copsimpleai/cpp_dataframes/'
    results_path = r'../copsimpleai/LocalSearch/Results/'
    run_file = r'../copsimpleai/LocalSearch/BestParamsPerAlgo/final_run.txt'
    LS_exe = r'../LocalSearch/Debug/LocalSearch.exe'
    path_best_args = r'../copsimpleai/LocalSearch/BestParamsPerAlgo/best_values_for_algs/'
    best_params_path = r'../copsimpleai/LocalSearch/BestParamsPerAlgo/'
    python_dataframes_path = r'../copsimpleai/python_dataframes/'

    run_time = 1.0
    num_iterations = 6  # used for optuna as number of trials, has to be accurate
    algo_seed = '331991908'
    # problem_set = ['291']
    problem_set = ['2656', '2701', '2734', '2869', '3118', '3223', '3258']
    # problem_set = ['2656', '2701', '2734', '2869', '3118', '3223', '3258', '3272', '3434', '3487', '3690',
    #        '3786', '3791', '4160', '4233', '4273', '4326', '4430', '4620', '4952']
    # problems = [str(random.randint(2500, 5000)) for _ in range(20)]
    # problem_set = problems.copy()
    # problem_set = ['271', '291', '375', '390', '500', '504', '549', '567', '643', '805', '1101', '1125', '2923', '3562']
    # algorithms = ['GREAT_DELUGE', 'SLBS', 'RS', 'RW', 'SHC', 'GREEDY', 'TS', 'SA', 'CE',
    #               'GREEDY+GREAT_DELUGE', 'GREEDY+SLBS', 'GREEDY+RS', 'GREEDY+RW', 'GREEDY+SHC', 'GREEDYLOOP',
    #               'GREEDY+TS', 'GREEDY+SA', 'GREEDY+CE']
    num_workers = 4

    # problem_set = ['271', '291']
    algorithms = ['SLBS', 'SHC', 'SA']
    COP_automized_run = COPLocalSearchAlgorithmsAveraging(problemGenerator_exe_path=problemCreatorPath,
                                                          results_path=results_path,
                                                          run_file_path=run_file,
                                                          LS_exe=LS_exe,
                                                          best_params_path=best_params_path,
                                                          best_val_for_alg_path=path_best_args,
                                                          algorithms=algorithms,
                                                          problem_seeds=problem_set,
                                                          algo_seed=algo_seed,
                                                          num_iterations=num_iterations,
                                                          run_time=run_time,
                                                          graphs_path=graphs_path,
                                                          cpp_dataframes_path=cpp_dataframes_path,
                                                          python_dataframes_path=python_dataframes_path,
                                                          num_workers=num_workers,
                                                          backup=True)

    # COP_automized_run.generate_problems_from_seeds()
    # COP_automized_run.run_optuna_param_optimization()  # run this if first you want to know the optimal params
    # COP_automized_run.find_best_params_run_then_output_expected_graphs(print_graphs_bool=False, ran_optimal_params=False)  # if optimal params already exist run this

    # python_results_path = r'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\Results\\'
    # python_run_file = r'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\BestParamsPerAlgo\python_final_run.txt'
    # python_exe = r'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\allAlgsInterface.py'
    # python_path_best_args = r'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\BestParamsPerAlgo\best_values_for_algs\\'
    # python_best_params_path = r'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\BestParamsPerAlgo\\'
    # # python_dataframes_path = r'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\python_dataframes\\'

    python_results_path = r'../copsimpleai/Results/'
    python_run_file = r'../copsimpleai/BestParamsPerAlgo/python_final_run.txt'
    python_exe = r'../copsimpleai/allAlgsInterface.py'
    python_path_best_args = r'../copsimpleai/BestParamsPerAlgo/best_values_for_algs/'
    python_best_params_path = r'../copsimpleai/BestParamsPerAlgo/'

    python_run_time = 15.0
    python_num_iterations = 6
    python_algo_seed = '331991908'
    # python_problem_set = ['291']
    #
    python_problem_set = ['2656', '2701', '2734', '2869', '3118', '3223', '3258']

    # python_problem_set = ['2656', '2701', '2734', '2869', '3118', '3223', '3258', '3272', '3434', '3487', '3690',
    #        '3786', '3791', '4160', '4233', '4273', '4326', '4430', '4620', '4952']
    # python_problem_set = ['271', '291', '375', '390', '500', '504', '549', '567', '643', '805', '1101', '1125', '2923', '3562']
    python_algs = ['LBS', 'SHC', 'SA']
    python_num_workers = 4

    COP_automized_run = COPLocalSearchAlgorithmsAveraging(problemGenerator_exe_path=problemCreatorPath,
                                                          results_path=python_results_path,
                                                          run_file_path=python_run_file,
                                                          LS_exe=python_exe,
                                                          best_params_path=python_best_params_path,
                                                          best_val_for_alg_path=python_path_best_args,
                                                          algorithms=python_algs,
                                                          problem_seeds=python_problem_set,
                                                          algo_seed=python_algo_seed,
                                                          num_iterations=python_num_iterations,
                                                          run_time=python_run_time,
                                                          graphs_path=graphs_path,
                                                          cpp_dataframes_path=cpp_dataframes_path,
                                                          python_dataframes_path=python_dataframes_path,
                                                          python=True,
                                                          num_workers=python_num_workers,
                                                          backup=True)

    COP_automized_run.run_optuna_param_optimization()
    COP_automized_run.find_best_params_run_then_output_expected_graphs(print_graphs_bool=False, ran_optimal_params=False)  # if optimal params already exist run this
    COP_automized_run.combined_expected_graphs()