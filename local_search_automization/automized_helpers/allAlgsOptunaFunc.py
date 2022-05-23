import subprocess


def run_optuna(trial, path, algo, problem_seed, algo_seed, python, run_time):
    """Executing a single run of an algorithm for some given problem seed.
    this run is essentially a trial run in order to find the future optimal parameters.

    Args:
        trial (int): trial number executed.
        algo (str): algorithm name executed.
        problem_seed (int): given problem seed.
        algo_seed (int): stochastic behavior of an algorithm.
        python (bool): True if running python algorithms from SimpleAi
            False if running CPP algorithms from LocalSearch.
        run_time (float): given run time for algorithm.

    Returns:
         float: Quality of the algorithm for the given trial (scalarization value).
    """
    algorithm = algo
    algorithm_seed = algo_seed
    problem_seed = problem_seed
    run_time = run_time
    neighborhood = trial.suggest_int('neighborhood', 3, 40)
    numelites = trial.suggest_int('numelites', 5, 15)
    inittemp = trial.suggest_float('inittemp', 10, 40)
    tempstep = trial.suggest_float('tempstep', 1, 10)
    tabusize = trial.suggest_int('tabusize', 5, 10)
    samples = trial.suggest_int('samples', 5, 20)
    initsolweight = trial.suggest_float('initsolweight', 0.0, 1.0)
    alpha = trial.suggest_float('alpha', 0.3, 0.9)
    rho = trial.suggest_float('rho', 0.5, 1.0)  # elite size
    epsilon = trial.suggest_float('epsilon', 0.2, 1.0)

    if python:
        x = subprocess.run(['python', path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                               , '-neighborhood', str(neighborhood), '-inittemp', str(inittemp), '-tempstep'
                               , str(tempstep), '-numelites', str(numelites), '-tabusize', str(tabusize)
                               , '-samples', str(samples), '-initsolweight', str(initsolweight), '-alpha', str(alpha)
                               , '-rho', str(rho), '-epsilon', str(epsilon)]
                               , capture_output=True, text=True)
    else:
        x = subprocess.run([path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                               , '-neighborhood', str(neighborhood), '-inittemp', str(inittemp), '-tempstep'
                               , str(tempstep), '-numelites', str(numelites), '-tabusize', str(tabusize)
                               , '-samples', str(samples), '-initsolweight', str(initsolweight), '-alpha', str(alpha)
                               , '-rho', str(rho), '-epsilon', str(epsilon)]
                               , capture_output=True, text=True)

    return float(str(x).split(',')[-3])  # magic to get the scalarization result


def simulated_annehiling_optuna(trial, path, algo, problem_seed, algo_seed, python, run_time):
    """Executing a single run of an algorithm for some given problem seed.
    this run is essentially a trial run in order to find the future optimal parameters.

    Args:
        trial (int): trial number executed.
        algo (str): algorithm name executed.
        problem_seed (int): given problem seed.
        algo_seed (int): stochastic behavior of an algorithm.
        python (bool): True if running python algorithms from SimpleAi
            False if running CPP algorithms from LocalSearch.
        run_time (float): given run time for algorithm.

    Returns:
         float: Quality of the algorithm for the given trial (scalarization value).
    """
    algorithm = algo
    algorithm_seed = algo_seed
    problem_seed = problem_seed
    run_time = run_time
    neighborhood = trial.suggest_int('neighborhood', 3, 40)
    inittemp = trial.suggest_float('inittemp', 10, 40)
    tempstep = trial.suggest_float('tempstep', 1, 10)

    if python:
        x = subprocess.run(['python', path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                               , '-neighborhood', str(neighborhood), '-inittemp', str(inittemp), '-tempstep'
                               , str(tempstep)]
                               , capture_output=True, text=True)
    else:
        x = subprocess.run([path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                               , '-neighborhood', str(neighborhood), '-inittemp', str(inittemp), '-tempstep'
                               , str(tempstep)]
                               , capture_output=True, text=True)

    return float(str(x).split(',')[-3])  # magic to get the scalarization result


def stochastic_local_beam_search_optuna(trial, path, algo, problem_seed, algo_seed, python, run_time):
    """Executing a single run of an algorithm for some given problem seed.
    this run is essentially a trial run in order to find the future optimal parameters.

    Args:
        trial (int): trial number executed.
        algo (str): algorithm name executed.
        problem_seed (int): given problem seed.
        algo_seed (int): stochastic behavior of an algorithm.
        python (bool): True if running python algorithms from SimpleAi
            False if running CPP algorithms from LocalSearch.
        run_time (float): given run time for algorithm.

    Returns:
         float: Quality of the algorithm for the given trial (scalarization value).
    """
    algorithm = algo
    algorithm_seed = algo_seed
    problem_seed = problem_seed
    run_time = run_time
    neighborhood = trial.suggest_int('neighborhood', 3, 40)
    numelites = trial.suggest_int('numelites', 5, 15)

    if python:
        x = subprocess.run(['python', path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                               , '-neighborhood', str(neighborhood), '-numelites', str(numelites)]
                               , capture_output=True, text=True)
    else:
        x = subprocess.run([path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                               , '-neighborhood', str(neighborhood),'-numelites', str(numelites)]
                               , capture_output=True, text=True)

    return float(str(x).split(',')[-3])  # magic to get the scalarization result


def random_search_or_greedy_optuna(trial, path, algo, problem_seed, algo_seed, python, run_time):
    """Executing a single run of an algorithm for some given problem seed.
    this run is essentially a trial run in order to find the future optimal parameters.

    Args:
        trial (int): trial number executed.
        algo (str): algorithm name executed.
        problem_seed (int): given problem seed.
        algo_seed (int): stochastic behavior of an algorithm.
        python (bool): True if running python algorithms from SimpleAi
            False if running CPP algorithms from LocalSearch.
        run_time (float): given run time for algorithm.

    Returns:
         float: Quality of the algorithm for the given trial (scalarization value).
    """
    algorithm = algo
    algorithm_seed = algo_seed
    problem_seed = problem_seed
    run_time = run_time

    if python:
        x = subprocess.run(['python', path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)]
                               , capture_output=True, text=True)
    else:
        x = subprocess.run([path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)]
                               , capture_output=True, text=True)

    return float(str(x).split(',')[-3])  # magic to get the scalarization result


def random_walk_optuna(trial, path, algo, problem_seed, algo_seed, python, run_time):
    """Executing a single run of an algorithm for some given problem seed.
    this run is essentially a trial run in order to find the future optimal parameters.

    Args:
        trial (int): trial number executed.
        algo (str): algorithm name executed.
        problem_seed (int): given problem seed.
        algo_seed (int): stochastic behavior of an algorithm.
        python (bool): True if running python algorithms from SimpleAi
            False if running CPP algorithms from LocalSearch.
        run_time (float): given run time for algorithm.

    Returns:
         float: Quality of the algorithm for the given trial (scalarization value).
    """
    algorithm = algo
    algorithm_seed = algo_seed
    problem_seed = problem_seed
    run_time = run_time
    neighborhood = trial.suggest_int('neighborhood', 3, 40)

    if python:
        x = subprocess.run(['python', path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                               , '-neighborhood', str(neighborhood)]
                               , capture_output=True, text=True)
    else:
        x = subprocess.run([path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                               , '-neighborhood', str(neighborhood)]
                               , capture_output=True, text=True)

    return float(str(x).split(',')[-3])  # magic to get the scalarization result


def stochastic_hill_climb_optuna(trial, path, algo, problem_seed, algo_seed, python, run_time):
    """Executing a single run of an algorithm for some given problem seed.
    this run is essentially a trial run in order to find the future optimal parameters.

    Args:
        trial (int): trial number executed.
        algo (str): algorithm name executed.
        problem_seed (int): given problem seed.
        algo_seed (int): stochastic behavior of an algorithm.
        python (bool): True if running python algorithms from SimpleAi
            False if running CPP algorithms from LocalSearch.
        run_time (float): given run time for algorithm.

    Returns:
         float: Quality of the algorithm for the given trial (scalarization value).
    """
    algorithm = algo
    algorithm_seed = algo_seed
    problem_seed = problem_seed
    run_time = run_time
    neighborhood = trial.suggest_int('neighborhood', 3, 40)

    if python:
        x = subprocess.run(['python', path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                               , '-neighborhood', str(neighborhood)]
                               , capture_output=True, text=True)
    else:
        x = subprocess.run([path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                               , '-neighborhood', str(neighborhood)]
                               , capture_output=True, text=True)

    return float(str(x).split(',')[-3])  # magic to get the scalarization result


def tabu_search_optuna(trial, path, algo, problem_seed, algo_seed, python, run_time):
    """Executing a single run of an algorithm for some given problem seed.
    this run is essentially a trial run in order to find the future optimal parameters.

    Args:
        trial (int): trial number executed.
        algo (str): algorithm name executed.
        problem_seed (int): given problem seed.
        algo_seed (int): stochastic behavior of an algorithm.
        python (bool): True if running python algorithms from SimpleAi
            False if running CPP algorithms from LocalSearch.
        run_time (float): given run time for algorithm.

    Returns:
         float: Quality of the algorithm for the given trial (scalarization value).
    """
    algorithm = algo
    algorithm_seed = algo_seed
    problem_seed = problem_seed
    run_time = run_time
    neighborhood = trial.suggest_int('neighborhood', 3, 40)
    tabusize = trial.suggest_int('tabusize', 5, 10)

    if python:
        x = subprocess.run(['python', path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                               , '-neighborhood', str(neighborhood),'-tabusize', str(tabusize)]
                               , capture_output=True, text=True)
    else:
        x = subprocess.run([path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                               , '-neighborhood', str(neighborhood), '-tabusize', str(tabusize)]
                               , capture_output=True, text=True)

    return float(str(x).split(',')[-3])  # magic to get the scalarization result


def great_deluge_optuna(trial, path, algo, problem_seed, algo_seed, python, run_time):
    """Executing a single run of an algorithm for some given problem seed.
    this run is essentially a trial run in order to find the future optimal parameters.

    Args:
        trial (int): trial number executed.
        algo (str): algorithm name executed.
        problem_seed (int): given problem seed.
        algo_seed (int): stochastic behavior of an algorithm.
        python (bool): True if running python algorithms from SimpleAi
            False if running CPP algorithms from LocalSearch.
        run_time (float): given run time for algorithm.

    Returns:
         float: Quality of the algorithm for the given trial (scalarization value).
    """
    algorithm = algo
    algorithm_seed = algo_seed
    problem_seed = problem_seed
    run_time = run_time
    neighborhood = trial.suggest_int('neighborhood', 3, 40)

    if python:
        x = subprocess.run(['python', path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                               , '-neighborhood', str(neighborhood)]
                               , capture_output=True, text=True)
    else:
        x = subprocess.run([path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                               , '-neighborhood', str(neighborhood)]
                               , capture_output=True, text=True)

    return float(str(x).split(',')[-3])  # magic to get the scalarization result


def cross_entropy_optuna(trial, path, algo, problem_seed, algo_seed, python, run_time):
    """Executing a single run of an algorithm for some given problem seed.
    this run is essentially a trial run in order to find the future optimal parameters.

    Args:
        trial (int): trial number executed.
        algo (str): algorithm name executed.
        problem_seed (int): given problem seed.
        algo_seed (int): stochastic behavior of an algorithm.
        python (bool): True if running python algorithms from SimpleAi
            False if running CPP algorithms from LocalSearch.
        run_time (float): given run time for algorithm.

    Returns:
         float: Quality of the algorithm for the given trial (scalarization value).
    """
    algorithm = algo
    algorithm_seed = algo_seed
    problem_seed = problem_seed
    run_time = run_time
    samples = trial.suggest_int('samples', 5, 20)
    initsolweight = trial.suggest_float('initsolweight', 0.1, 0.5)
    alpha = trial.suggest_float('alpha', 0.3, 0.5)
    rho = trial.suggest_float('rho', 0.5, 0.8)  # elite size
    epsilon = trial.suggest_float('epsilon', 0.2, 0.8)

    if python:
        x = subprocess.run(['python', path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                               , '-samples', str(samples), '-initsolweight', str(initsolweight), '-alpha', str(alpha)
                               , '-rho', str(rho), '-epsilon', str(epsilon)]
                               , capture_output=True, text=True)
    else:
        x = subprocess.run([path, algorithm, str(problem_seed), str(run_time), str(algorithm_seed)
                               , '-samples', str(samples), '-initsolweight', str(initsolweight), '-alpha', str(alpha)
                               , '-rho', str(rho), '-epsilon', str(epsilon)]
                               , capture_output=True, text=True)

    return float(str(x).split(',')[-3])  # magic to get the scalarization result
