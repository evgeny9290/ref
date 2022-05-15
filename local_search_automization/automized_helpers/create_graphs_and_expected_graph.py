import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np
from copy import deepcopy


def fill_arrays_with_file_names_initial(path):
    """Fills 6 arrays with the appropriate file names to iterate upon later.

    Args:
        path (str): path to the folder which contains the results of every algorithm.

    Returns:
         (list[str], list[str], list[str], list[str], list[str], list[str]):
            6-tuple containing 6 arrays where each contains all files with appropriate name.
    """
    initial_greedy_best_val_files = []
    with_greedy_best_vals_files = []
    best_val_files = []

    initial_greedy_time_files = []
    with_greedy_time_files = []
    time_files = []

    for file in os.listdir(path):
        if 'BestValue' in file and '+' in file and 'Initial' not in file:
            with_greedy_best_vals_files.append(file)
        if 'BestValue' in file and '+' not in file:
            best_val_files.append(file)
        if 'BestValue' in file and '+' in file and 'Initial' in file:
            initial_greedy_best_val_files.append(file)

        if 'Times' in file and '+' in file and 'Initial' not in file:
            with_greedy_time_files.append(file)
        if 'Times' in file and '+' not in file:
            time_files.append(file)
        if 'Times' in file and '+' in file and 'Initial' in file:
            initial_greedy_time_files.append(file)

    return initial_greedy_best_val_files, with_greedy_best_vals_files, best_val_files \
        , initial_greedy_time_files, with_greedy_time_files, time_files


def remove_unused_files(path):
    """Removing unused and unneeded files from path.

    Args:
        path (str): path to the folder which contains the results of every algorithm.

    Returns:
         None.
    """
    for file in os.listdir(path):
        if 'Current' in file:
            os.remove(path + file)


def merge_files_into_initial(best_vals_files, initial_best_val_files, time_files, initial_time_files, path):
    """If there are algorithms that run greedy first and there after some other type of algorithm
    they will create two separate files with the results. One with the greedyInit and another with the other algorithm.
    This function will merging the files content into a single file such that the latter algorithm is on top
    of the other file content.
    Afterwards deleting all unnecessary files (those that include with greedy_init)

    Args:
        best_vals_files (list[str]): values array of file names such that the algorithm ran AFTER InitGreedy.
        initial_best_val_files (list[str]): values array of file names such that the algorithm ran InitGreedy.
        time_files (list[str]): times array of file names such that the algorithm ran AFTER InitGreedy.
        initial_time_files (list[str]): times array of file names such that the algorithm ran InitGreedy.
        path (str): path to the folder which contains the results of every algorithm.

    Returns:
         None.
    """
    for file_read, file_write in zip(best_vals_files, initial_best_val_files):
        name1 = file_read
        name2 = file_write
        with open(path + name1, "r") as file:
            data = file.read()
        with open(path + name2, "a") as fout:
            fout.write(data)

    for file_read, file_write in zip(time_files, initial_time_files):
        name1 = file_read
        name2 = file_write
        with open(path + name1, "r") as file:
            data2 = file.read()

        with open(path + name2, "a+") as fout:
            fout.seek(0)
            cur_data = fout.read()
            try:
                max_time = float(cur_data.split('\n')[-1])
            except ValueError:
                max_time = float(cur_data.split('\n')[-2])
            except Exception as e:
                print(e)
            data2 = "\n".join(list(map(lambda x: str(float(x) + max_time), data2.split('\n')[:-1])))
            fout.write(data2)

    for file in best_vals_files:
        os.remove(path + file)

    for file in time_files:
        os.remove(path + file)


def scalarizarion(line):
    """Scalarize the GradeVector given as a parameter.

    Args:
        line (list[float]): GradeVector to be scalarized.

    Returns:
        float: scalarization value of the given GradeVector.
    """
    res = 0
    multiplier = 1
    for val in line[::-1]:
        if val == 0:
            continue
        res += val * multiplier
        multiplier *= 200
    return res


def scalarize_all(arr):
    """Scalarize 2d array by calling "scalarizarion" for every entry.

    Args:
        arr (list[list[float]]): 2d array of GradeVectors such that every entry is a GradeVector converted into an array.

    Returns:
         list[float]: array of scalarized values.
    """
    for i, entry in enumerate(arr):
        arr[i] = scalarizarion(entry)
    return arr


def scalarized_best_values_all_files(best_value_files, path):
    """Scalarize everything by calling the previous functions "scalarize_all" and "scalarizarion"
    for every file in "best_value_files"

    Args:
        best_value_files (list[str]): final file names after with all correct GradeVector values in each row.
        path (str): path to the folder which contains the results of every algorithm.

    Returns:
         list[list[float]]: 2d array such that every entry is already scalarized accordingly.
    """
    results = []
    for file in best_value_files:
        res = []
        with open(path + file, 'r') as f:
            for line in f:
                formated_line = [float(x) for x in line.split()]
                res.append(formated_line)
        results.append(res)

    for idx, algo in enumerate(results):
        results[idx] = scalarize_all(algo)

    return results


def times_all_files(time_files, path):
    """Converting each line in every file into float (as time should be float) and after converting appending
    each file into an array. After appending every single file the result is 2d array.

    Args:
        time_files (list[str]): array of time file names
        path (str): path to the folder which contains the results of every algorithm.

    Returns:
        list[list[float]]: 2d array of floats such that the rows are the file names and columns are the algorithm times.
    """
    results = []
    for file in time_files:
        res = []
        with open(path + file, 'r') as f:
            for line in f:
                res.append(float(line))
        results.append(res)

    return results


def abs_scalarization(all_algos_best_vals_scalarized):
    """After Scalarization convert every entry into its abs form, making the value positive.

    Args:
        all_algos_best_vals_scalarized (list[list[float]]): 2d array such that every entry is already scalarized accordingly.

    Returns:
         list[list[float]]: abs(on input)
    """
    all_algos_best_vals_scalarized = [[abs(x) for x in algo] for algo in all_algos_best_vals_scalarized]
    return all_algos_best_vals_scalarized


def extract_names(best_value_files, time_files):
    """Extract the names of the algorithms from the arrays with file names as input.

    Args:
        best_value_files (list[str]): best_value array with file names
        time_files (list[str]): time array with file names

    Returns:
         (list[str], list[str]): array with algorithm names both for values and times.
    """
    best_val_algo_names, time_algo_names = [], []
    for val, time in zip(best_value_files, time_files):
        if 'GREAT_DELUGE' in val:
            best_val_title = '_'.join(val.split('_')[:5])
            time_title = '_'.join(time.split('_')[:5])
        else:
            best_val_title = '_'.join(val.split('_')[:4])
            time_title = '_'.join(time.split('_')[:4])

        best_val_algo_names.append(best_val_title)
        time_algo_names.append(time_title)

    return best_val_algo_names, time_algo_names


def create_data_frames(all_algos_best_vals_scalarized, all_algos_times, best_val_names, time_names):
    """Creating DataFrames from the arrays passed as input.
    every entry from "all_algos_best_vals_scalarized" and "all_algos_times" will result in a series
    and every entry from "best_val_names" and "time_names" will result as a header(column) in the DataFrame.

    Args:
        all_algos_best_vals_scalarized (list[list[float]]): 2d array of scalarized values for all problems all algorithms.
        all_algos_times (list[list[float]]): 2d array of time values for all problems all algorithms.
        best_val_names (list[str]): best_value array with file names.
        time_names (list[str]): time array with file names.

    Returns:
        list[DataFrame]: array of DataFrames with cols [best_val_name, time_name].
    """
    results = []
    for best_val, time, best_val_name, time_name in zip(all_algos_best_vals_scalarized, all_algos_times, best_val_names, time_names):
        results.append(pd.DataFrame({best_val_name: best_val, time_name: time}))
    return results


def problem_algo_matrix(algs_df_arr, num_problems, num_algos):
    """Transforming the array input into a 2d array where the rows will be the problem seeds casted as index
    and cols will be the algorithm DataFrame for that "problem seed" casted as index.

    Args:
        algs_df_arr (list[DataFrame]): array of DataFrames
        num_problems (int): number of problems.
        num_algos (int): number of algorithms.
    Returns:
         list[list[DataFrame]]: 2d array where the rows are the problem number (casted as index)
            and cols will be the algorithm DataFrame for that problem (casted as index).

    Notes:
        all_probs_algs[i][j]: will be DataFrame for algorithm numbered j for problem numbered i.
    """
    all_probs_algs = []
    for problem in range(num_problems):
        algos_for_problem = []
        for algo in range(num_algos):
            algos_for_problem.append(algs_df_arr[problem + algo * num_problems])
        all_probs_algs.append(algos_for_problem)
    return all_probs_algs


def insert_row_if_early_finish(all_probs_algs, run_time):
    """Some Algorithms do not run for their whole "run_time", some iteration might not in the given time.
    Hence We manually append one row into each DataFrame that has "finished early" to smoothen the graph.

    Args:
        all_probs_algs (list[list[DataFrame]]): 2d array where the rows are the problem number (casted as index)
            and cols will be the algorithm DataFrame for that problem (casted as index).

    Returns:
         list[list[DataFrame]]: same as input but with an additional row if "finished early".
    """
    for p_idx, problem in enumerate(all_probs_algs):
        for idx, df in enumerate(problem):
            col1, col2 = df.columns[0], df.columns[1]
            if df.iloc[-1][1] < run_time:
                new_row = {col1: df.iloc[-1][0], col2: run_time}
                all_probs_algs[p_idx][idx] = pd.concat([all_probs_algs[p_idx][idx],
                                                        pd.DataFrame(new_row, index=[0])],
                                                       ignore_index=True)
            # FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version
                # all_probs_algs[p_idx][idx] = df.append(new_row, ignore_index=True)
    return all_probs_algs


def graph_for_all_problems(all_probs_algs, problem_seeds, python, graphs_path, algo_seed):
    """Creates a graph for every problem in "problem_seeds" with all algorithms
     using the DataFrames from "all_probs_algs"

    Args:
        all_probs_algs (list[list[DataFrame]]): 2d array of DataFrames.
        problem_seeds (list[str]): array of problem seeds.
        python (bool): True if running python algorithms from SimpleAi
                False if running CPP algorithms from LocalSearch.
        graphs_path (str): path to the folder which will contain the graph problem.

    Returns:
        None.
    """
    sns.set(rc={'figure.figsize': (20.7, 12.27)})

    for problem, problem_seed in zip(all_probs_algs,problem_seeds):
        for df in problem:
            col1, col2 = df.columns[0], df.columns[1]
            name_arr = col1.split('_')
            if name_arr[1] == 'GREAT':
                label = '_'.join(name_arr[1:3])
            else:
                label = name_arr[1]
            if '+' in label or 'LOOP' in label:
                plt.plot(df[col2], df[col1], label=label, linestyle='-.', linewidth=1.5)
            else:
                plt.plot(df[col2], df[col1], label=label, linestyle='--', linewidth=1.5)
            plt.xlabel('time in seconds', fontsize=18)
            plt.ylabel('quality', fontsize=18, rotation='horizontal')
            if python:
                plt.title(f'Python problem seed: {problem_seed}\n algo seed: {algo_seed}', fontsize=18)
            else:
                plt.title(f'CPP problem seed: {problem_seed}\n algo seed: {algo_seed}', fontsize=18)
        plt.legend(loc='center left', bbox_to_anchor=(0.96, 0.5))
        if python:
            plt.savefig(graphs_path + fr'python_problem_num_{problem_seed}.png')
            plt.savefig(graphs_path + fr'python_problem_num_{problem_seed}.pdf')
        else:
            plt.savefig(graphs_path + fr'CPP_problem_num_{problem_seed}.png')
            plt.savefig(graphs_path + fr'CPP_problem_num_{problem_seed}.pdf')

        plt.clf()


# prep for expected graph funcs

def prob_algo_len_matrix(all_probs_algs, num_problems, num_algos):
    """Calculates the length (number of rows) for every DataFrames passed in "all_probs_algs."

    Args:
        all_probs_algs (list[list[DataFrame]]): 2d array of DataFrames.
        num_problems (int): number of problems.
        num_algos (int): number of algorithms.

    Returns:
        list[list[int]]: number of rows of every DataFrame for every problem.
    """
    len_dfs = []
    for problem in range(num_problems):
        prob = []
        for algo in range(num_algos):
            prob.append(len(all_probs_algs[problem][algo]))
        len_dfs.append(prob)

    return len_dfs


def max_len_df_foreach_algo(len_dfs, num_problems, num_algos):
    """Calculating the maximum number of rows for the same algorithm in each problem.

    Args:
        len_dfs (list[list[int]]): number of rows of every DataFrame for every problem.
        num_problems (int): number of problems.
        num_algos (int): number of algorithms.

    Returns:
        list[int]: maximum number of rows for same algorithm all problems.
    """
    max_len_df_per_algo = []
    for algo in range(num_algos):
        max_len_for_algo = -1
        for problem in range(num_problems):
            max_len_for_algo = max(max_len_for_algo, len_dfs[problem][algo])
        max_len_df_per_algo.append(max_len_for_algo)

    return max_len_df_per_algo


def algo_for_prob_matrix(all_probs_algs, num_problems, num_algos):
    """Essentially a Transpose, now the rows are the algorithms and cols are the problems.

    Args:
        all_probs_algs (list[list[DataFrame]]): 2d array of DataFrames.
        num_problems (int): number of problems.
        num_algos (int): number of algorithms.

    Returns:
        list[list[DataFrame]]: 2d array where the rows are the algorithms and cols are the problems.
    """
    same_algs_all_probs = [[all_probs_algs[prob][alg] for prob in range(num_problems)] for alg in range(num_algos)]
    return same_algs_all_probs


def fix_dim_all_probs_all_algs_inplace(same_algs_all_probs, max_len_df_per_algo, python,
                                       cpp_dataframes_path, python_dataframes_path, backup, run_time):
    """Reshaping the DataFrames to be of the same shape in order to find the expected DataFrame.
    Also creating backup DataFrames in appropriate path if backup wanted.

    Args:
        same_algs_all_probs (list[list[DataFrame]]): 2d array where the rows are the algorithms and cols are the problems.
        max_len_df_per_algo (list[int]): maximum number of rows for same alg all problems.
        python (bool): True if running python algorithms from SimpleAi
                False if running CPP algorithms from LocalSearch.
        cpp_dataframes_path (str): path to where the backup DataFrames will saved to if comes from cpp.
        python_dataframes_path (str): path to where the backup DataFrames will saved to if comes from python.
        backup (bool): True if should backup DataFrames else False.

    Returns:
        None.
    """
    for idx_alg , (alg, max_len_for_algo) in enumerate(zip(same_algs_all_probs, max_len_df_per_algo)):
        for idx_df, df in enumerate(alg):
            len_to_add = max_len_for_algo - len(df)
            new_series_col1 = [df.iloc[-1][0]] * len_to_add
            # new_series_col2 = [1] * len_to_add  # run_time
            new_series_col2 = [run_time] * len_to_add
            new_df = pd.DataFrame({df.columns[0]: new_series_col1, df.columns[1]: new_series_col2})
            same_algs_all_probs[idx_alg][idx_df] = pd.concat([df, new_df], ignore_index=True)
            if backup:
                if python:
                    same_algs_all_probs[idx_alg][idx_df].to_csv(
                        python_dataframes_path + fr'python_df_{df.columns[0]}_{idx_df}.csv',
                        index=False)
                else:
                    same_algs_all_probs[idx_alg][idx_df].to_csv(
                        cpp_dataframes_path + fr'cpp_df_{df.columns[0]}_{idx_df}.csv',
                        index=False)


def create_expected_dfs_all_algs(same_dim_benchmark):
    """Creating expected DataFrames for every problem.

    Args:
        same_dim_benchmark (list[list[DataFrame]]): 2d array of DataFrames with the same shape.

    Returns:
         list[DataFrame]: array of expected DataFrames.
    """
    expected_df_all_algs = []
    for alg_idx, alg in enumerate(same_dim_benchmark):
        df_temp = alg[0]
        for df in alg[1:]:
            df_temp += df.values
        df_temp /= len(alg)
        expected_df_all_algs.append(df_temp)

    return expected_df_all_algs

# this function is not usefull at this moment. for now i leave it blank for future reference.
def time_smoothing_for_expected_dfs(extra_backup_after_normalize, max_len_df_per_algo, run_time):
    """Smoothing the time column of each expected DataFrame so there wont be sudden jumps and
    to make the graph monotonically increasing.

    Args:
        extra_backup_after_normalize (list[DataFrame]): array of expected DataFrames.
        max_len_df_per_algo (list[int]): maximum number of rows for same alg all problems.
        run_time (float): run time of an algorithm

    Return:
        None.
    """
    # for algo, max_len in zip(extra_backup_after_normalize, max_len_df_per_algo):
    #     col_time = algo.columns[1]
    #     try:
    #         algo[col_time] = np.arange(0, run_time, run_time / max_len)
    #     except:
    #         algo[col_time] = np.arange(0, run_time, run_time / max_len)[:-1]


def rename_df_cols_inplace(same_dim_benchmark_after_normalize_fixed_time):
    """Renaming the column headers for the Expected DataFrames inplace.

    Args:
        same_dim_benchmark_after_normalize_fixed_time (list[DataFrame]): array of expected DataFrames.

    Returns:
         None.
    """
    for df in same_dim_benchmark_after_normalize_fixed_time:
        col1, col2 = df.columns[0], df.columns[1]
        new_col1_name = "_".join(col1.split('_')[:-2]) + '_Expected'
        new_col2_name = "_".join(col2.split('_')[:-2]) + '_Expected'
        df.rename(columns={col1: new_col1_name, col2: new_col2_name}, inplace=True)


def expected_dfs_to_csv_inpalce(same_dim_benchmark_after_normalize_fixed_time, python, cpp_dataframes_path, python_dataframes_path):
    """Creating expected DataFrames .csv files in dataframes_path location according to python flag.

    Args:
        same_dim_benchmark_after_normalize_fixed_time (list[DataFrame]): array of expected DataFrames.
        python (bool): True if running python algorithms from SimpleAi
                False if running CPP algorithms from LocalSearch.
        cpp_dataframes_path (str): path to where the backup DataFrames will saved to if comes from cpp.
        python_dataframes_path (str): path to where the backup DataFrames will saved to if comes from python.

    Returns:
         None.
    """
    for expected_df in same_dim_benchmark_after_normalize_fixed_time:
        if python:
            expected_df.to_csv(python_dataframes_path + fr'expected_dataframes/python_expected_df{expected_df.columns[0]}.csv', index=False)
        else:
            expected_df.to_csv(cpp_dataframes_path + fr'expected_dataframes/cpp_expected_df{expected_df.columns[0]}.csv', index=False)


def expected_graph_all_algs(same_dim_benchmark_after_normalize_fixed_time, python, graphs_path, algo_seed):
    """Create expected graph for all algorithms all problems.

    Args:
        same_dim_benchmark_after_normalize_fixed_time (list[DataFrame]): array of expected DataFrames.
        python (bool): True if running python algorithms from SimpleAi
                    False if running CPP algorithms from LocalSearch.
        graphs_path (str): path to the folder which will contain the expected graph.

    Returns:
        None.
    """
    sns.set(rc={'figure.figsize': (20.7, 12.27)})
    for df in same_dim_benchmark_after_normalize_fixed_time:
        col1, col2 = df.columns[0], df.columns[1]
        name_arr = col1.split('_')
        if name_arr[1] == 'GREAT':
            label = '_'.join(name_arr[1:3])
        else:
            label = name_arr[1]
        if '+' in col1:
            plt.plot(df[col2], df[col1], label=label, linestyle='-.', linewidth=1.5)
        elif '+' and 'LOOP' in col1:
            plt.plot(df[col2], df[col1], label=label, linestyle=':', linewidth=1.5)
        else:
            plt.plot(df[col2], df[col1], label=label, linestyle='--', linewidth=1.5)
        plt.xlabel('time in seconds', fontsize=18)
        plt.ylabel('quality', fontsize=18, rotation='horizontal')
        if python:
            plt.title(f'Python Expected Graph\n algo seed: {algo_seed}', fontsize=18)
        else:
            plt.title(f'CPP Expected Graph\n algo seed: {algo_seed}', fontsize=18)

    plt.legend(loc='center left', bbox_to_anchor=(0.96, 0.5))
    if python:
        plt.savefig(graphs_path + fr'Python_Expected_Graph.png')
        plt.savefig(graphs_path + fr'Python_Expected_Graph.pdf')

    else:
        plt.savefig(graphs_path + fr'CPP_Expected_Graph.png')
        plt.savefig(graphs_path + fr'CPP_Expected_Graph.pdf')

    plt.clf()


def sanity_check_file_names(best_val_files, time_files):
    """For Debugging, check if the file names are the same after the first '_'.
    printing the files which do not have the same name.

    Args:
        best_val_files (list[str]): best values array of file names.
        time_files (list[str]): times array of file names.

    Returns:
         None.
    """
    for val_file, time_file in zip(best_val_files, time_files):
        check_one = "_".join(val_file.split('_')[1:])
        check_two = "_".join(time_file.split('_')[1:])
        if check_one != check_two:
            print(val_file, time_file)


def automize_graphs_per_algo(path, num_algos, num_problems, problem_seeds, print_all_graphs, python, graphs_path, run_time, algo_seed):
    """Automating whole process until graph creation for all problems.
    calling helper functions in correct order.

    Args:
        path (str): path to the folder which contains the results of every algorithm.
        num_algos (int): number of algorithms.
        num_problems (int): number of problems.
        problem_seeds (list[str]): array of problem seeds.
        print_all_graphs (bool): True if want to print all graphs per problem else False.
        python (bool): True if running python algorithms from SimpleAi
                    False if running CPP algorithms from LocalSearch.
        graphs_path (str): path to the folder which will contain the expected graph.
        run_time (float): run time of an algorithm

    Returns:
        list[list[DataFrame]]: 2d array where the rows are the problem number (casted as index)
            and cols will be the algorithm DataFrame for that problem (casted as index).
    """
    initial_greedy_best_val_files, with_greedy_best_vals_files, best_val_files, initial_greedy_time_files, with_greedy_time_files, time_files = fill_arrays_with_file_names_initial(path)
    remove_unused_files(path)
    merge_files_into_initial(with_greedy_best_vals_files, initial_greedy_best_val_files, with_greedy_time_files, initial_greedy_time_files, path)

    best_value_files_final = initial_greedy_best_val_files + best_val_files
    time_files_final = initial_greedy_time_files + time_files

    sanity_check_file_names(best_value_files_final, time_files_final)
    all_algos_best_vals_scalarized = scalarized_best_values_all_files(best_value_files_final, path)
    all_algos_times = times_all_files(time_files_final, path)
    all_algos_best_vals_scalarized = abs_scalarization(all_algos_best_vals_scalarized)
    best_val_names, time_names = extract_names(best_value_files_final, time_files_final)
    algs_df_arr = create_data_frames(all_algos_best_vals_scalarized, all_algos_times, best_val_names, time_names)
    all_probs_algs = problem_algo_matrix(algs_df_arr, num_problems, num_algos)
    all_probs_algs = insert_row_if_early_finish(all_probs_algs, run_time)

    if print_all_graphs:
        graph_for_all_problems(all_probs_algs, problem_seeds, python, graphs_path, algo_seed)

    return all_probs_algs


def automize_expected_graph(path, num_algos, num_problems, problem_seeds, print_all_graphs,
                            graphs_path, python=False, run_time=1.0,
                            cpp_dataframes_path=".", python_dataframes_path=".", backup=False, algo_seed=0):
    """Automating whole process until expected graph creation.
    calling helper functions in correct order.

    Args:
        path (str): path to the folder which contains the results of every algorithm.
        num_algos (int): number of algorithms.
        num_problems (int): number of problems.
        problem_seeds (list[str]): array of problem seeds.
        print_all_graphs (bool): True if want to print all graphs per problem else False.
        graphs_path (str): path to the folder which will contain the expected graph.
        python (bool): True if running python algorithms from SimpleAi
            False if running CPP algorithms from LocalSearch.
        run_time (float): run time of an algorithm
        cpp_dataframes_path (str): path to where the backup DataFrames will saved to if comes from cpp.
        python_dataframes_path (str): path to where the backup DataFrames will saved to if comes from python.
        backup (bool): True if want to same all DataFrames as .csv in appropriate path else False.

    Returns:
        None.
    """
    all_probs_algs = automize_graphs_per_algo(path, num_algos, num_problems, problem_seeds, print_all_graphs, python, graphs_path, run_time, algo_seed)
    len_dfs = prob_algo_len_matrix(all_probs_algs, num_problems, num_algos)
    max_len_df_per_algo = max_len_df_foreach_algo(len_dfs, num_problems, num_algos)
    same_algs_all_probs = algo_for_prob_matrix(all_probs_algs, num_problems, num_algos)
    fix_dim_all_probs_all_algs_inplace(same_algs_all_probs, max_len_df_per_algo, python, cpp_dataframes_path, python_dataframes_path, backup, run_time)
    same_dim_benchmark = deepcopy(same_algs_all_probs)
    expected_df_all_algs = create_expected_dfs_all_algs(same_dim_benchmark)
    # same_dim_benchmark_after_normalize = deepcopy(expected_df_all_algs)  # not really needed, was here as a benchmark for debugging.
    time_smoothing_for_expected_dfs(expected_df_all_algs, max_len_df_per_algo, run_time)  # same_dim_benchmark_after_normalize
    same_dim_benchmark_after_normalize_fixed_time = deepcopy(expected_df_all_algs) # same_dim_benchmark_after_normalize
    rename_df_cols_inplace(same_dim_benchmark_after_normalize_fixed_time)
    expected_dfs_to_csv_inpalce(same_dim_benchmark_after_normalize_fixed_time, python, cpp_dataframes_path, python_dataframes_path)

    expected_graph_all_algs(same_dim_benchmark_after_normalize_fixed_time, python, graphs_path, algo_seed)

