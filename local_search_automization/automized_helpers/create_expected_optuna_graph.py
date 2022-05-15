import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def read_best_vals_files_to_array(path):
    """Reads files names into array

    Args:
        path (str): path to folder containing files with best_values so far from Optuna.

    Returns:
        list[str]: array containing the file names.
    """
    best_val_all_algos = []
    for file in os.listdir(path):
        best_val_all_algos.append(file)
    return best_val_all_algos


def convert_to_dfs(best_val_all_algos, path):
    """Converts files content into DataFrames with pandas and inserting them in a list.

    Args:
        best_val_all_algos (list[str]): array containing the file names.
        path (str): path to folder containing files with best_values so far from Optuna.

    Returns:
        list[DataFrame]: array of DataFrames from files located at path.
    """
    dfs = []
    # for file in os.listdir(path):
    for file in best_val_all_algos:
        if '.txt' in file:
            df = pd.read_csv(path + file, header=None)
            if 'GREAT' in file:
                if '+' in file:
                    df.rename(columns={0: 'Best_Vals_' + "_".join(file.split('_')[2:6]) + '.txt'}, inplace=True)
                else:
                    df.rename(columns={0: 'Best_Vals_' + "_".join(file.split('_')[2:6])}, inplace=True)
            else:
                if '+' in file:
                    df.rename(columns={0: 'Best_Vals_' + "_".join(file.split('_')[2:5]) + '.txt'}, inplace=True)
                else:
                    df.rename(columns={0: 'Best_Vals_' + "_".join(file.split('_')[2:5])}, inplace=True)
            dfs.append(df)
    return dfs


def all_probs_one_alg(all_algs_optuna_dfs, num_algs, num_problems):
    """Reconstructing the array structure into 2d array, such that:
    arr[alg][prob] = algorithm alg for problem prob

    Args:
        all_algs_optuna_dfs (list[DataFrame]): array of DataFrames where each DataFrame corresponds to some algorithm and problem.
        num_algs (int): number of algorithms
        num_problems (int): number of problems

    Returns:
        list[list[DataFrame]]: 2d array such that each entry is a DataFrame
            where the rows are algorithms and cols are problems.
    """
    all_problems_optuna_one_alg = [[all_algs_optuna_dfs[i+j*num_problems] for i in range(num_problems)] for j in range(num_algs)]
    return all_problems_optuna_one_alg


def create_expected_df_all_algs_arr(all_problems_optuna_one_alg, num_iterations):
    """Creates an array of DataFrames where each DataFrame is the expectation
    of all the DataFrames with the same Algorithm for all problems.
    result DataFrame columns: [BestValue, iterations] where BestValue are positive.

    Args:
        all_problems_optuna_one_alg (list[list[DataFrame]]): 2d array such that each entry is a DataFrame
            where the rows are algorithms and cols are problems.
        num_iterations (int): number of iterations which Optuna ran for.

    Returns:
        list[DataFrame]: array of expected DataFrames from all the algorithms for all problems.

    """
    expected_df_all_algs = []
    for alg_idx, alg in enumerate(all_problems_optuna_one_alg):
        df_temp = alg[0]
        for df in alg[1:]:
            df_temp += df.values
        df_temp /= len(alg)
        expected_df_all_algs.append(df_temp)

    for algo in expected_df_all_algs:
        algo['iterations'] = np.arange(1, num_iterations + 1, 1)

    for algo in expected_df_all_algs:
        algo[algo.columns[0]] = algo[algo.columns[0]].apply(abs)

    return expected_df_all_algs


def expected_graph_optuna_parametrization(expected_df_all_algs, python, graphs_path, algo_seed):
    """Creates expected Optuna monotonic increasing graph in "graphs_path" location.

    expected_df_all_algs (list[DataFrame]): array of expected DataFrames from all the algorithms for all problems.
    python (bool): True if running python algorithms from SimpleAi
            False if running CPP algorithms from LocalSearch.
    graphs_path (str): path to the folder which will contain the Expected Optuna Graph.

    Returns:
         None
    """
    sns.set(rc={'figure.figsize': (20.7, 12.27)})

    for df in expected_df_all_algs:
        col1, col2 = df.columns[0], df.columns[1]
        name_arr = col1.split('_')
        if name_arr[2] == 'GREAT':
            label = '_'.join(name_arr[2:3])
        else:
            label = name_arr[2]
        if '+' in label:
            plt.plot(df[col2], df[col1], label=label, linestyle='-.', linewidth=1.5)
        elif '+' and 'LOOP' in label:
            plt.plot(df[col2], df[col1], label=label, linestyle=':', linewidth=1.5)
        else:
            plt.plot(df[col2], df[col1], label=label, linestyle='--', linewidth=1.5)
        plt.xlabel('iterations', fontsize=18)
        plt.ylabel('best_quality', fontsize=18, rotation='horizontal', loc='center')
        if python:
            plt.title(f'Python Expected Graph Optuna\n algo seed: {algo_seed}', fontsize=18)
        else:
            plt.title(f'CPP Expected Graph Optuna\n algo seed: {algo_seed}', fontsize=18)
    plt.legend(loc='center left', bbox_to_anchor=(0.96, 0.5))
    if python:
        plt.savefig(graphs_path + fr'Python_Expected_Graph_Optuna_all_algs.png')
        plt.savefig(graphs_path + fr'Python_Expected_Graph_Optuna_all_algs.pdf')

    else:
        plt.savefig(graphs_path + fr'CPP_Expected_Graph_Optuna_all_algs.png')
        plt.savefig(graphs_path + fr'CPP_Expected_Graph_Optuna_all_algs.pdf')

    plt.clf()


def automize_optuna_expected_graph(path, num_algs, num_problems, num_iterations, graphs_path, python=False, algo_seed=0):
    """Automizing whole process of creating the Expected Optuna Graph.
    Calling all helper functions in corrent order.

    Args:
        path (str): path to folder containing files with best_values so far from Optuna.
        num_algs (int): number of algorithms
        num_problems (int): number of problems
        num_iterations (int): number of iterations which Optuna ran for.
        graphs_path (str): path to the folder which will contain the Expected Optuna Graph.
        python (bool): True if running python algorithms from SimpleAi
            False if running CPP algorithms from LocalSearch.

    Returns:
        None.
    """
    best_val_all_algos = read_best_vals_files_to_array(path)
    all_algs_optuna_dfs = convert_to_dfs(best_val_all_algos, path)
    all_problems_optuna_one_alg = all_probs_one_alg(all_algs_optuna_dfs, num_algs, num_problems)
    expected_df_all_algs = create_expected_df_all_algs_arr(all_problems_optuna_one_alg, num_iterations)
    expected_graph_optuna_parametrization(expected_df_all_algs, python, graphs_path, algo_seed)
