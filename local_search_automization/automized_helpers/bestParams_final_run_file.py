import os
import re


def read_files_to_arrays(target_location):
    """Extracts file names into two seperate arrays.
    If Greedy_Init included in that name, appends to best_params_init_greedy array
    Else appends to best_params array

    Args:
        target_location: path to the folder which has all file problems with best parameters
            for each problem seed for each algorithm

    Returns:
        tuple(list[str], list[str]): array of file names without greedy, array of file names with greedy
    """
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
    """If there are algorithms that run greedy first and there after some other type of algorithm
    they will create two separate files with the results. One with the greedyInit and another with the other algorithm.
    This function will merging the files content into a single file such that the latter algorithm is on top
    of the other file content.
    Afterwards deleting all unnecessary files (those that include with greedy_init)

    Args:
        best_params (list[str]): array of files names without greedy in them
        best_params_init_greedy (list[str]): array of files names with greedy in them
        path: path to the folder which has all file problems with best parameters
            for each problem seed for each algorithm

    Returns:
        None

    Notes:
        Currently this function is useless and not needed anymore (it was here for testing)
        function will work only if the array length's are the same, meaning if you run an algorithm
        and its greedy_init counterpart.
    """
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
    """Converting refactored content of files into a 2d array.
    First extracting all the problem seeds numbers from the file names and appending them into num_array.
    Reconstructing their full names, appending the correct refactored format into a 2d array where each row is
    a list containing all algorithms for specific problem seed and all cols are the algorithms for that seed.

    Args:
        files (list[str]): array containing all the file names
        target_location (str): path to the folder which has all file problems with best parameters
            for each problem seed for each algorithm
    Returns:
         list[list[str]]: 2d array where the cols are all the refactored parameters for each algorithm
            and rows are all the problems.

    Notes:
        output example:
            all_params =
            [[params for algorithm "SA" problem number x, params for algorithm "SHC" problem number x, ...],
             [params for algorithm "SA" problem number y, params for algorithm "SHC" problem number y, ...],
             [params for algorithm "SA" problem number z, params for algorithm "SHC" problem number z, ...]]
    """
    all_params = []  # should be 2d array later on
    num_array = []  # problem seed numbers
    for file in os.listdir(target_location):
        if '.txt' in file:
            generic_name = file
            get_number = generic_name.split('_')[2][:-4]
            num_array.append(get_number)
    num_array = sorted(num_array, key=lambda x: int(x))

    files = [files[0][:19] + prob + '.txt' for prob in num_array]  # full file names reconstruct
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
    """Converts files into a 2d array containing outputs as in "Notes" section.
    Args:
        all_params list[list[str]]: 2d array where the cols are all the refactored parameters for each algorithm
            and rows are all the problems.
    Returns:
         list[list[str]]: 2d array where the cols are all the refactored parameters for each algorithm
            and rows are all the problems.

    Notes:
        This function was used for debugging, maybe need to delete.
        output example:
            output =
            [[params for algorithm "SA" problem number x, params for algorithm "SHC" problem number x, ...],
             [params for algorithm "SA" problem number y, params for algorithm "SHC" problem number y, ...],
             [params for algorithm "SA" problem number z, params for algorithm "SHC" problem number z, ...]]
    """
    output = []
    for idx, file in enumerate(all_params):
        f = [file[algo] for algo in range(len(all_params[idx]))]
        output.append(f)
    return output


def convert_all_to_3d_list(all_params):
    """Converting a 2d array into 3d array such that
    arr[i][j] = all parameters for a specific algorithm for a specific problem.
    arr[i][j][k] = parameter for a specific algorithm for a specific problem.

    Args:
        all_params (list[list[str]]): 2d array where the cols are all the refactored parameters for each algorithm
                and rows are all the problems.
    Returns:
        list[list[list[str]]: 3d array, same as 2d array from last function but including "split" hence transforming
            the inside string into an array splitted by "," creating another dim.

    Notes:
        res =
        [[[params for algorithm "SA" problem number x], [params for algorithm "SHC" problem number x], ...]],
         [[params for algorithm "SA" problem number y], [params for algorithm "SHC" problem number y], ...]],
         [[params for algorithm "SA" problem number z], [params for algorithm "SHC" problem number z], ...]]]
    """
    res = []
    for prob_idx, file_num in enumerate(all_params):
        temp = []
        for algo_idx, params in enumerate(all_params[prob_idx]):
            temp.append(params.split(','))
        res.append(temp)
    return res


def insert_problem_seed(final_list, problem_set, run_time, algo_seed):
    """Inserts problem_seed, run_time, algo_seed in appropriate places for Optuna to run on later.
    Args:
        final_list (list[list[list[str]]): arr[i][j] = all parameters for a specific algorithm for a specific problem.
            arr[i][j][k] = parameter for a specific algorithm for a specific problem.
        problem_set (list[str]): array of problem seed's.
        run_time (float): run time parameter for every algorithm.
        algo_seed (str): seed for stochastic behaviour of algorithm.

    Returns:
         list[list[list[str]]: 3d array, same as 2d array from last function but including "split" hence transforming
            the inside string into an array splitted by "," creating another dim.
    """
    for file, problem in zip(final_list, problem_set):
        for algo in file:
            algo.insert(1, problem)
            algo.insert(2, str(run_time))
            algo.insert(3, str(algo_seed))
    return final_list


def clean_all_files(final_list):
    """Refactors the parameters into the final form such that the result format is appropriate for running
    and being equivalent to the format ran in the terminal.

    Args:
        final_list (list[list[list[str]]): 3d array, same as 2d array from last function but including "split" hence transforming
            the inside string into an array splitted by "," creating another dim.
    Returns:
         list[list[list[str]]: same as input but refactored appropriately.
    """
    new_res = []
    args = ['neighborhood', 'numelites', 'inittemp', 'tempstep', 'tabusize', 'samples', 'initsolweight', 'alpha', 'rho',
            'epsilon']
    for problem in final_list:
        temp_prob = []
        for algo in problem:
            temp_algo = []
            for word in algo:
                word = re.sub(r"[\']", "", word.strip())
                if word in args:
                    word = '-' + word

                temp_algo.append(word)
            temp_prob.append(temp_algo)
        new_res.append(temp_prob)
    return new_res


def write_clean_res_to_file(clean_res, target_location, python):
    """Creating "final run" .txt file including all optimal parameters for every algorithm for every problem.

    Args:
        clean_res (list[list[list[str]]): cleaned and has the desired format in 3d array and ready to run.
        target_location (str): path to the folder which has all file problems with best parameters
            for each problem seed for each algorithm
        python (bool):  True if running python algorithms from SimpleAi
                False if running CPP algorithms from LocalSearch.

    Returns:
         None.
    """
    target_name = target_location + 'python_final_run.txt' if python else target_location + 'final_run.txt'
    with open(target_name, "w") as f:
        for file in clean_res:
            for algo in file:
                f.write((",").join(algo) + '\n')


def automize_final_run_file(target_location, problem_set, algo_seed, python=False, run_time='1.0'):
    """Automizing the process of creating "final run" .txt file with optimal parameters achieved by running Optuna.

    Args:
        target_location (str): path to the folder which has all file problems with best parameters
            for each problem seed for each algorithm
        problem_set (list[str]): array of problem seed's.
        algo_seed (str): seed for stochastic behaviour of algorithm.
        python (bool):  True if running python algorithms from SimpleAi
                False if running CPP algorithms from LocalSearch.
        run_time (float): run time parameter for every algorithm.

    Returns:
        None
    """
    best_params, best_params_init_greedy = read_files_to_arrays(target_location)
    merge_files_into_initial(best_params, best_params_init_greedy, target_location)
    all_params = read_all_files(best_params, target_location)
    params_per_problem_set = convert_to_list_of_files(all_params)  # function not needed
    final_list = convert_all_to_3d_list(params_per_problem_set)
    final_list = insert_problem_seed(final_list, problem_set, run_time, algo_seed)
    final_list = clean_all_files(final_list)
    write_clean_res_to_file(final_list, target_location, python)
