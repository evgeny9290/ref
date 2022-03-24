import os
import pandas as pd
from copy import deepcopy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def read_best_vals_files_to_array(path):
    best_val_all_algos = []
    for file in os.listdir(path):
        best_val_all_algos.append(file)
    return best_val_all_algos


def convert_to_dfs(best_val_all_algos, path):
    dfs = []
    for file in os.listdir(path):
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
    all_problems_optuna_one_alg = [[all_algs_optuna_dfs[i+j*num_problems] for i in range(num_problems)] for j in range(num_algs)]
    return all_problems_optuna_one_alg


def create_expected_df_all_algs_arr(all_problems_optuna_one_alg, num_iterations):
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


def expected_graph_optuna_parametrization(expected_df_all_algs, python):
    sns.set(rc={'figure.figsize':(20.7,12.27)})

    for df in expected_df_all_algs:
        col1, col2 = df.columns[0], df.columns[1]
        name_arr = col1.split('_')
        if name_arr[2] == 'GREAT':
            label = ('_').join(name_arr[2:3])
        else:
            label = name_arr[2]
        if '+' in label:
            plt.plot(df[col2],df[col1], label=label, linestyle='-.', linewidth=1.5)
        elif 'LOOP' in label:
            plt.plot(df[col2],df[col1], label=label, linestyle=':', linewidth=1.5)
        else:
            plt.plot(df[col2],df[col1], label=label, linestyle='--', linewidth=1.5)
        plt.xlabel('iterations', fontsize=18)
        plt.ylabel('best_quality', fontsize=18, rotation='horizontal',loc='center')
        plt.title('Expected Graph Optuna', fontsize=18)
    plt.legend(loc='center left', bbox_to_anchor=(0.96, 0.5))
    if python:
        plt.savefig(fr'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\graphs\Python_Expected_Graph_Optuna_all_algs.png')
    else:
        plt.savefig(fr'graphs\Expected_Graph_Optuna_all_algs.png')
    plt.clf()


def automize_optuna_expected_graph(path, num_algs, num_problems, num_iterations, python=False):
    best_val_all_algos = read_best_vals_files_to_array(path)
    all_algs_optuna_dfs = convert_to_dfs(best_val_all_algos, path)
    all_problems_optuna_one_alg = all_probs_one_alg(all_algs_optuna_dfs, num_algs, num_problems)
    expected_df_all_algs = create_expected_df_all_algs_arr(all_problems_optuna_one_alg, num_iterations)
    expected_graph_optuna_parametrization(expected_df_all_algs, python)


if __name__ == '__main__':
    # path = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\BestParamsPerAlgo\best_values_for_algs\\'
    # num_algos = 18
    # num_problems = 14
    # num_iterations = 20
    #
    # automize_optuna_expected_graph(path, num_algos, num_problems, num_iterations)

    python_path = r'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\python_output\best_values_for_algs\\'
    python_num_algos = 3
    python_num_problems = 14
    python_num_iterations = 8

    automize_optuna_expected_graph(python_path, python_num_algos, python_num_problems, python_num_iterations, python=True)