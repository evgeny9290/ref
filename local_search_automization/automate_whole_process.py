import os

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
                 problem_seeds, algo_seed, num_iterations, alg_run_time, optuna_run_time, graphs_path,
                 cpp_dataframes_path=".", python_dataframes_path=".",
                 python=False, num_workers=1, backup=False, unique_trials=False):
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
            alg_run_time (float): run time of optimal parameter for every algorithm.
            optuna_run_time (float): run time for optuna parametrization for every algorithm.
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
        self.alg_run_time = alg_run_time
        self.optuna_run_time = optuna_run_time
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
        self.unique_trials = unique_trials

    def create_final_run_and_run_algs_from_final_run(self, ran_optimal_params):  # both create final_run file if doesnt exist and run prog with those params
        """If file doesnt exist Creates "final run" .txt file where each line contains the optimal parameters found for a specific algorithm via Optuna.
         If didn't run commands from "final run" .txt file first execute them.

        Args:
            ran_optimal_params (bool): True if already executed algorithms with optimal parameters, else False.

        Returns:
             None.
        """
        if not os.path.exists(self.run_file_path):
            automize_final_run_file(self.best_params_path, self.problem_seeds, self.algo_seed, self.python, self.alg_run_time)
        if not ran_optimal_params:
            run_algs_with_optimal_params(self.run_file_path, self.LS_exe, self.python, self.num_workers, self.algo_seed)

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
        try:
            os.makedirs(self.best_val_for_alg_path + '../..')
        except FileExistsError as e:
            pass

        best_params_for_all_algos(self.LS_exe, self.best_params_path, self.algorithms,
                                  self.problem_seeds, self.algo_seed, self.num_iterations, self.python, self.optuna_run_time, self.unique_trials)

    def create_expected_graphs(self, print_graphs_bool):  # only expected_graphs and graphs for each problem if needed
        """Calls upon functions responsible for Creating expected graphs, both for Optuna and all Algorithms.

        Args:
            print_graphs_bool (bool): True if want to print all graphs for every problem else False.

        Returns:
            None.
        """
        try:
            os.makedirs(self.graphs_path)
            os.makedirs(self.cpp_dataframes_path + "expected_dataframes")
            os.makedirs(self.python_dataframes_path + "expected_dataframes")
            print("directories created successfully")
        except OSError as e:
            print("something went wrong with directory creation")
            print(e)

        automize_optuna_expected_graph(self.best_val_for_alg_path, self.num_algos, self.num_problems,
                                       self.num_iterations, self.graphs_path, self.python, self.algo_seed)
        automize_expected_graph(self.results_path, self.num_algos, self.num_problems,
                                self.problem_seeds, print_graphs_bool, self.graphs_path, self.python, self.alg_run_time,
                                self.cpp_dataframes_path, self.python_dataframes_path, self.backup, self.algo_seed)

    def create_final_run_file(self):  # only final_run file
        """Creates "final run" .txt file where each line contains the optimal parameters found for a specific algorithm via Optuna.

        Args:
            None.

        Returns:
            None.
        """
        automize_final_run_file(self.best_params_path, self.problem_seeds, self.algo_seed, self.python, self.alg_run_time)

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
        automize_combined_expected_graphs(self.cpp_dataframes_path + f"expected_dataframes/",
                                          self.python_dataframes_path + f"expected_dataframes/",
                                          self.graphs_path, self.algo_seed)

if __name__ == '__main__':
    # algo_seeds = ['199190833', '331991908', '222991908']
    algo_seeds = ['199190833']

    for algo_seed in algo_seeds:
        problemCreatorPath = r'../LocalSearchProblemGenerator/Debug/LocalSearchProblemGenerator.exe'
        graphs_path = rf'../copsimpleai/graphs/algo_seed_{algo_seed}/'
        cpp_dataframes_path = rf'../copsimpleai/cpp_dataframes/algo_seed_{algo_seed}/'
        results_path = rf'../copsimpleai/LocalSearch/Results/'
        run_file = rf'../copsimpleai/LocalSearch/algo_seed_{algo_seed}/BestParamsPerAlgo/final_run.txt'
        LS_exe = r'../LocalSearch/Debug/LocalSearch.exe'
        path_best_args = rf'../copsimpleai/LocalSearch/algo_seed_{algo_seed}/BestParamsPerAlgo/best_values_for_algs/'
        best_params_path = rf'../copsimpleai/LocalSearch/algo_seed_{algo_seed}/BestParamsPerAlgo/'
        python_dataframes_path = rf'../copsimpleai/python_dataframes/algo_seed_{algo_seed}/'

        alg_run_time = 120.0
        optuna_run_time = 2.0
        num_iterations = 24  # used for optuna as number of trials, has to be accurate

        problem_set = ['3118', '641233', '632142']  # small test
        # problem_set = ['632142']  # small test

        # problem_set = ['231231']

        # problem_set = ['2701', '2734', '3118', '3487', '3690', '4620', '4952']  # medium test

        # problem_set = ['2656', '2701', '2734', '2869', '3118', '3223', '3258', '3272', '3434', '3487', '3690',
        #                '3786', '3791', '4160', '4233', '4273', '4326', '4430', '4620', '4952']  # big test
        # problems = [str(random.randint(2500, 5000)) for _ in range(20)]  # can generate rand problems this way
        # problem_set = problems.copy()

        # algorithms = ['SLBS', 'RS', 'RW', 'SHC', 'GREEDY', 'TS', 'SA', 'CE',
        #               'GREEDY+SLBS', 'GREEDY+RS', 'GREEDY+RW', 'GREEDY+SHC', 'GREEDYLOOP',
        #               'GREEDY+TS', 'GREEDY+SA', 'GREEDY+CE']
        num_workers = 5

        algorithms = ['RS', 'RW', 'SHC', 'SA', 'GREEDY', 'GREEDYLOOP',
                      'GREEDY+SA', 'GREEDY+SHC', 'GREEDY+RW', 'GREEDY+RS']
        # algorithms = ['GREEDY+SHC', 'GREEDY', 'SHC', 'SA']
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
                                                              alg_run_time=alg_run_time,
                                                              optuna_run_time=optuna_run_time,
                                                              graphs_path=graphs_path,
                                                              cpp_dataframes_path=cpp_dataframes_path,
                                                              python_dataframes_path=python_dataframes_path,
                                                              python=False,
                                                              num_workers=num_workers,
                                                              backup=True,
                                                              unique_trials=False)

        COP_automized_run.generate_problems_from_seeds()
        # COP_automized_run.run_optuna_param_optimization()  # run this if first you want to know the optimal params
        # COP_automized_run.find_best_params_run_then_output_expected_graphs(print_graphs_bool=True, ran_optimal_params=True)  # if optimal params already exist run this

        python_results_path = r'../copsimpleai/pythonLocalSearch/Results/'
        python_run_file = fr'../copsimpleai/pythonLocalSearch/algo_seed_{algo_seed}/BestParamsPerAlgo/python_final_run.txt'
        python_exe = r'../copsimpleai/allAlgsInterface.py'
        python_path_best_args = fr'../copsimpleai/pythonLocalSearch/algo_seed_{algo_seed}/BestParamsPerAlgo/best_values_for_algs/'
        python_best_params_path = fr'../copsimpleai/pythonLocalSearch/algo_seed_{algo_seed}/BestParamsPerAlgo/'

        python_alg_run_time = 120.0
        python_optuna_run_time = 60.0
        python_num_iterations = 24
        python_algo_seed = algo_seed

        python_problem_set = ['3118', '641233', '632142']  # small test
        # python_problem_set = ['632142']  # medium test


        # python_problem_set = ['2656', '2701', '2734', '2869', '3118', '3223', '3258', '4233', '4273', '4326']  # medium test

        # python_problem_set = ['2656', '2701', '2734', '2869', '3118', '3223', '3258', '3272', '3434', '3487', '3690',
        #                       '3786', '3791', '4160', '4233', '4273', '4326', '4430', '4620', '4952']  # big test
        # python_algs = ['GREEDY+LOOP']
        python_algs = ['GREEDY+LOOP', 'RS', 'RW', 'SHC', 'SA', 'GREEDY',
                      'GREEDY+SA', 'GREEDY+SHC', 'GREEDY+RW', 'GREEDY+RS']


        python_num_workers = 5

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
                                                              alg_run_time=python_alg_run_time,
                                                              optuna_run_time=python_optuna_run_time,
                                                              graphs_path=graphs_path,
                                                              cpp_dataframes_path=cpp_dataframes_path,
                                                              python_dataframes_path=python_dataframes_path,
                                                              python=True,
                                                              num_workers=python_num_workers,
                                                              backup=True,
                                                              unique_trials=False)

        COP_automized_run.run_optuna_param_optimization()
        COP_automized_run.find_best_params_run_then_output_expected_graphs(print_graphs_bool=True, ran_optimal_params=False)  # if optimal params already exist run this
        COP_automized_run.combined_expected_graphs()
