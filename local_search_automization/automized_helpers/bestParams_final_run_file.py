import os
import pandas as pd
import re


def read_files_to_arrays(target_location):
    best_params = []
    best_params_init_greedy = []
    for file in os.listdir(target_location):
        if '.txt' in file:
            if 'Greedy_Init.txt' in file:
                best_params_init_greedy.append(file)
            else:
                best_params.append(file)
    return best_params, best_params_init_greedy


def merge_files_into_initial(best_params, best_params_init_greedy, path):
    if len(best_params) != len(best_params_init_greedy):
        return

    for file_read, file_write in zip(best_params_init_greedy, best_params):
        name1 = file_read
        name2 = file_write
        with open(path + name1, "r") as file:
            data2 = file.read()
        with open(path + name2, "a") as fout:
            fout.write(data2)

    for file in best_params_init_greedy:
        os.remove(path + file)


def read_all_files(files, target_location):
    all_params = []
    num_array = []
    for file in os.listdir(target_location):
        if '.txt' in file:
            generic_name = file
            get_number = generic_name.split('_')[2][:-4]
            num_array.append(get_number)
    num_array = sorted(num_array, key=lambda x: int(x))

    files = [files[0][:19] + prob + '.txt' for prob in num_array]
    for file in files:
        params = []
        with open(target_location + file) as f:
            for line in f:
                line = re.sub(r"[{}\n]", "", line)
                line = re.sub(r"[:]", ",", line)
                params.append(line)
        all_params.append(params)

    return all_params


def convert_to_list_of_files(all_params):
    output = []
    for idx, file in enumerate(all_params):
        f = [file[algo] for algo in range(len(all_params[idx]))]
        output.append(f)
    return output


def convert_all_to_3d_list(all_params):
    res = []
    for prob_idx, file_num in enumerate(all_params):
        temp = []
        for algo_idx, params in enumerate(all_params[prob_idx]):
            temp.append(params.split(','))
        res.append(temp)
    return res


def insert_problem_seed(final_list, problem_set):
    run_time = '1.0'
    algo_seed = '331991908'
    for file, problem in zip(final_list,problem_set):
        for algo in file:
            algo.insert(1, problem)
            algo.insert(2, run_time)
            algo.insert(3, algo_seed)
    return final_list


def clean_all_files(final_list):
    new_res = []
    args = ['neighborhood', 'numelites', 'inittemp', 'tempstep', 'tabusize', 'samples', 'initsolweight', 'alpha', 'rho',
            'repsilon']
    for problem in final_list:
        temp_prob = []
        for algo in problem:
            temp_algo = []
            for word in algo:
                word = re.sub(r"[\']", "", word.strip())
                if word in args:
                    if word == 'repsilon':  # was a typo, next run remove 'r' from 'repsilon' and this if
                        word = 'epsilon'
                    word = '-' + word

                temp_algo.append(word)
            temp_prob.append(temp_algo)
        new_res.append(temp_prob)
    return new_res


def write_clean_res_to_file(clean_res, target_location):
    with open(target_location + 'final_run.txt', "w") as f:
        for file in clean_res:
            for algo in file:
                f.write((",").join(algo) + '\n')


def automize_final_run_file(target_location, problem_set):
    best_params, best_params_init_greedy = read_files_to_arrays(target_location)
    merge_files_into_initial(best_params, best_params_init_greedy, target_location)
    all_params = read_all_files(best_params, target_location)
    params_per_problem_set = convert_to_list_of_files(all_params)
    final_list = convert_all_to_3d_list(params_per_problem_set)
    final_list = insert_problem_seed(final_list, problem_set)
    final_list = clean_all_files(final_list)
    write_clean_res_to_file(final_list, target_location)


if __name__ == '__main__':
    target_location = r'C:\Users\evgni\Desktop\Projects\LocalSearch\LocalSearch\BestParamsPerAlgo\\'
    problem_set = ['182', '271', '291', '375', '390', '504', '549', '567', '643', '805', '1101', '1125', '2923', '3562']

    automize_final_run_file(target_location, problem_set)