import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np
from copy import deepcopy


def fill_arrays_with_file_names_initial(path):
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
    for file in os.listdir(path):
        if 'Current' in file:
            os.remove(path + file)


def merge_files_into_initial(best_vals_files, initial_best_val_files, time_files, initial_time_files, path):
    for file_read, file_write in zip(best_vals_files, initial_best_val_files):
        name1 = file_read
        name2 = file_write
        with open(path + name1, "r") as file:
            data2 = file.read()
        with open(path + name2, "a") as fout:
            fout.write(data2)

    for file_read, file_write in zip(time_files, initial_time_files):
        name1 = file_read
        name2 = file_write
        with open(path + name1, "r") as file:
            data2 = file.read()
        with open(path + name2, "a") as fout:
            fout.write(data2)

    for file in best_vals_files:
        os.remove(path + file)

    for file in time_files:
        os.remove(path + file)


def scalarizarion(line):
    res = 0
    multiplier = 1
    for val in line[::-1]:
        if val == 0:
            continue
        res += val * multiplier
        multiplier *= 200
    return res


def scalarize_all(arr):
    for i, entry in enumerate(arr):
        arr[i] = scalarizarion(entry)
    return arr


def scalarized_best_values_all_files(best_value_files, path):
    results = []
    for file in best_value_files:
        res = []
        with open(path + file, 'r') as f:
            for line in f:
                formated_line = [float(x) for x in line.split()]
                res.append(formated_line)
        results.append(res)

    for algo in results:
        scalarize_all(algo)

    return results


def times_all_files(time_files, path):
    results = []
    for file in time_files:
        res = []
        with open(path + file, 'r') as f:
            for line in f:
                res.append(float(line))
        results.append(res)

    return results


def abs_scalarization(all_algos_best_vals_scalarized):
    all_algos_best_vals_scalarized = [[abs(x) for x in algo] for algo in all_algos_best_vals_scalarized]
    return all_algos_best_vals_scalarized


def extract_names(best_value_files, time_files):
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
    results = []
    for best_val, time, best_val_name, time_name in zip(all_algos_best_vals_scalarized, all_algos_times, best_val_names, time_names):
        results.append(pd.DataFrame({best_val_name: best_val, time_name: time}))
    return results


def problem_algo_matrix(algs_df_arr, num_problems, num_algos):
    all_probs_algs = []
    for problem in range(num_problems):
        algos_for_problem = []
        for algo in range(num_algos):
            algos_for_problem.append(algs_df_arr[problem + algo * num_problems])
        all_probs_algs.append(algos_for_problem)
    return all_probs_algs


def insert_row_if_early_finish(all_probs_algs):
    for p_idx, problem in enumerate(all_probs_algs):
        for idx, df in enumerate(problem):
            col1, col2 = df.columns[0], df.columns[1]
            if df.iloc[-1][1] < 1:
                new_row = {col1:df.iloc[-1][0], col2:1}
                all_probs_algs[p_idx][idx] = df.append(new_row, ignore_index=True)
    return all_probs_algs


def graph_for_all_problems(all_probs_algs, problem_seeds, python):
    sns.set(rc={'figure.figsize':(20.7,12.27)})

    for problem, problem_seed in zip(all_probs_algs,problem_seeds):
        for df in problem:
            col1, col2 = df.columns[0], df.columns[1]
            name_arr = col1.split('_')
            if name_arr[1] == 'GREAT':
                label = ('_').join(name_arr[1:3])
            else:
                label = name_arr[1]
            if '+' in label or 'LOOP' in label:
                plt.plot(df[col2],df[col1], label=label, linestyle='-.', linewidth=1.5)
            else:
                plt.plot(df[col2],df[col1], label=label, linestyle='--', linewidth=1.5)
            plt.xlabel('time in seconds', fontsize=18)
            plt.ylabel('quality', fontsize=18, rotation='horizontal')
            plt.title(f'problem seed: {problem_seed}', fontsize=18)
        plt.legend(loc='center left', bbox_to_anchor=(0.96, 0.5))
        if python:
            plt.savefig(fr'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\graphs\problem_num_{problem_seed}.png')
        else:
            plt.savefig(fr'graphs\problem_num_{problem_seed}.png')
        plt.clf()


# prep for expected graph funcs

def prob_algo_len_matrix(all_probs_algs, num_problems, num_algos):
    len_dfs = []
    for problem in range(num_problems):
        prob = []
        for algo in range(num_algos):
            prob.append(len(all_probs_algs[problem][algo]))
        len_dfs.append(prob)

    return len_dfs


def max_len_df_foreach_algo(len_dfs, num_problems, num_algos):
    max_len_df_per_algo = []
    for algo in range(num_algos):
        max_len_for_algo = -1
        for problem in range(num_problems):
            max_len_for_algo = max(max_len_for_algo, len_dfs[problem][algo])
        max_len_df_per_algo.append(max_len_for_algo)

    return max_len_df_per_algo


def algo_for_prob_matrix(all_probs_algs, num_problems, num_algos):
    same_algs_all_probs = [[all_probs_algs[prob][alg] for prob in range(num_problems)] for alg in range(num_algos)]
    return same_algs_all_probs


def fix_dim_all_probs_all_algs_inplace(same_algs_all_probs, max_len_df_per_algo):
    for idx_alg , (alg, max_len_for_algo) in enumerate(zip(same_algs_all_probs, max_len_df_per_algo)):
        # print(max_len_for_algo)
        for idx_df, df in enumerate(alg):
            # print(df)
            len_to_add = max_len_for_algo - len(df)
            new_series_col1 = [df.iloc[-1][0]] * len_to_add
            new_series_col2 = [1] * len_to_add
            new_df = pd.DataFrame({df.columns[0]: new_series_col1, df.columns[1]: new_series_col2})
            same_algs_all_probs[idx_alg][idx_df] = pd.concat([df, new_df], ignore_index=True)
            # print(same_algs_all_probs[idx_alg][idx_df])
            same_algs_all_probs[idx_alg][idx_df].to_csv(fr'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\dataframes\df_{df.columns[0]}_{idx_df}.csv', index=False)


def create_expected_dfs_all_algs(same_dim_benchmark):
    expected_df_all_algs = []
    for alg_idx, alg in enumerate(same_dim_benchmark):
        df_temp = alg[0]
        for df in alg[1:]:
            df_temp += df.values
        df_temp /= len(alg)
        expected_df_all_algs.append(df_temp)

    return expected_df_all_algs


def time_smoothing_for_expected_dfs(extra_backup_after_normalize, max_len_df_per_algo, run_time):
    for algo, max_len in zip(extra_backup_after_normalize, max_len_df_per_algo):
        col_time = algo.columns[1]
        try:
            algo[col_time] = np.arange(0, run_time, run_time / max_len)
        except:
            algo[col_time] = np.arange(0, run_time, run_time / max_len)[:-1]


def rename_df_cols_inplace(same_dim_benchmark_after_normalize_fixed_time):
    for df in same_dim_benchmark_after_normalize_fixed_time:
        col1, col2 = df.columns[0], df.columns[1]
        new_col1_name = "_".join(col1.split('_')[:-2]) + '_Expected'
        new_col2_name = "_".join(col2.split('_')[:-2]) + '_Expected'
        df.rename(columns={col1: new_col1_name, col2: new_col2_name}, inplace=True)


def expected_dfs_to_csv_inpalce(same_dim_benchmark_after_normalize_fixed_time, python):
    for expected_df in same_dim_benchmark_after_normalize_fixed_time:
        if python:
            expected_df.to_csv(fr'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\dataframes\expected_dataframes\expected_df{expected_df.columns[0]}.csv', index=False)
        else:
            expected_df.to_csv(fr'dataframes\expected_dataframes\expected_df{expected_df.columns[0]}.csv', index=False)


def expected_graph_all_algs(same_dim_benchmark_after_normalize_fixed_time, python):
    sns.set(rc={'figure.figsize':(20.7,12.27)})
    for df in same_dim_benchmark_after_normalize_fixed_time:
        col1, col2 = df.columns[0], df.columns[1]
        name_arr = col1.split('_')
        if name_arr[1] == 'GREAT':
            label = ('_').join(name_arr[1:3])
        else:
            label = name_arr[1]
        if '+' in col1:
            plt.plot(df[col2],df[col1], label=label, linestyle='-.', linewidth=1.5)
        elif 'LOOP' in col1:
            plt.plot(df[col2],df[col1], label=label, linestyle=':', linewidth=1.5)
        else:
            plt.plot(df[col2],df[col1], label=label, linestyle='--', linewidth=1.5)
        plt.xlabel('time in seconds', fontsize=18)
        plt.ylabel('quality', fontsize=18, rotation='horizontal')
        plt.title('Expected Graph', fontsize=18)
    plt.legend(loc='center left', bbox_to_anchor=(0.96, 0.5))
    if python:
        plt.savefig(fr'C:\Users\evgni\Desktop\projects_mine\ref\ref\copsimpleai\graphs\Python_Expected_Graph.png')
    else:
        plt.savefig(fr'graphs\Expected_Graph.png')
    plt.clf()


def sanity_check_file_names(best_val_files, time_files):
    for val_file, time_file in zip(best_val_files, time_files):
        check_one = "_".join(val_file.split('_')[1:])
        check_two = "_".join(time_file.split('_')[1:])
        if check_one != check_two:
            print(val_file, time_file)


def automize_graphs_per_algo(path, num_algos, num_problems, problem_seeds, print_all_graphs, python):
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
    all_probs_algs = insert_row_if_early_finish(all_probs_algs)

    if print_all_graphs:
        graph_for_all_problems(all_probs_algs, problem_seeds, python)

    return all_probs_algs


def automize_expected_graph(path, num_algos, num_problems, problem_seeds, print_all_graphs, python=False, run_time=1.0):
    all_probs_algs = automize_graphs_per_algo(path, num_algos, num_problems, problem_seeds, print_all_graphs, python)
    len_dfs = prob_algo_len_matrix(all_probs_algs, num_problems, num_algos)
    max_len_df_per_algo = max_len_df_foreach_algo(len_dfs, num_problems, num_algos)
    same_algs_all_probs = algo_for_prob_matrix(all_probs_algs, num_problems, num_algos)
    fix_dim_all_probs_all_algs_inplace(same_algs_all_probs, max_len_df_per_algo)
    same_dim_benchmark = deepcopy(same_algs_all_probs)
    expected_df_all_algs = create_expected_dfs_all_algs(same_dim_benchmark)
    same_dim_benchmark_after_normalize = deepcopy(expected_df_all_algs)
    time_smoothing_for_expected_dfs(same_dim_benchmark_after_normalize, max_len_df_per_algo, run_time)
    same_dim_benchmark_after_normalize_fixed_time = deepcopy(same_dim_benchmark_after_normalize)
    rename_df_cols_inplace(same_dim_benchmark_after_normalize_fixed_time)
    expected_dfs_to_csv_inpalce(same_dim_benchmark_after_normalize_fixed_time, python)

    expected_graph_all_algs(same_dim_benchmark_after_normalize_fixed_time, python)


if __name__ == '__main__':
    path = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\Results\\'
    num_algos = 18
    num_problems = 14
    problem_seeds = ['182', '271', '291', '375', '390', '504', '549', '567', '643', '805', '1101', '1125', '2923', '3562']

    automize_expected_graph(path, num_algos, num_problems, problem_seeds, False)
