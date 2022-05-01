# coding=utf-8
from simpleai.search.utils import BoundedPriorityQueue, InverseTransformSampler
from simpleai.search.models import SearchNodeValueOrdered
import math
import random
import datetime
import numpy as np
import os
from copy import deepcopy, copy
from copsimpleai.structClasses import SolutionVector, GradesVector
from copsimpleai.Constants import MAX_NUM_OF_VARS


def _all_expander(fringe, iteration, viewer, problem):
    '''
    Expander that expands all nodes on the fringe.
    '''
    expanded_neighbors = [node.expand(local_search=True)
                          for node in fringe]

    if viewer:
        viewer.event('expanded', list(fringe), expanded_neighbors)

    list(map(fringe.extend, expanded_neighbors))


def beam(problem, beam_size=100, iterations_limit=0, viewer=None):
    '''
    Beam search.

    beam_size is the size of the beam.
    If iterations_limit is specified, the algorithm will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.actions, SearchProblem.result, SearchProblem.value,
    and SearchProblem.generate_random_state.
    '''
    return _local_search(problem,
                         _all_expander,
                         iterations_limit=iterations_limit,
                         fringe_size=beam_size,
                         random_initial_states=True,
                         stop_when_no_better=iterations_limit==0,
                         viewer=viewer)


def _first_expander(fringe, iteration, viewer, problem):
    '''
    Expander that expands only the first node on the fringe.
    '''
    current = fringe[0]
    neighbors = current.expand(local_search=True)

    if viewer:
        viewer.event('expanded', [current], [neighbors])

    fringe.extend(neighbors)


def _random_walk_expander(curr_sol, iteration, viewer, problem):
    '''
    Expander that expands only the first node on the fringe.
    '''
    current = curr_sol
    neighbors = current.expand(local_search=True)

    if viewer:
        viewer.event('expanded', [current], [neighbors])

    return neighbors


def random_walk(problem, iterations_limit=0, viewer=None, max_run_time=1, seed=0):
    '''
    Random Walk.

    If iterations_limit is specified, the algorithm will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.actions, SearchProblem.result, and
    SearchProblem.value.
    '''
    random.seed(seed)
    return _local_search(problem,
                         _random_walk_expander,
                         iterations_limit=iterations_limit,
                         fringe_size=1,
                         stop_when_no_better=iterations_limit==0,
                         viewer=viewer,
                         max_run_time=max_run_time)


def beam_best_first(problem, beam_size=100, iterations_limit=0, viewer=None, max_run_time=1, seed=0):
    '''
    Beam search best first.

    beam_size is the size of the beam.
    If iterations_limit is specified, the algorithm will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.actions, SearchProblem.result, and
    SearchProblem.value.
    '''
    random.seed(seed)
    return _local_search(problem,
                         _first_expander,
                         iterations_limit=iterations_limit,
                         fringe_size=beam_size,
                         random_initial_states=True,
                         stop_when_no_better=iterations_limit==0,
                         viewer=viewer,
                         max_run_time=max_run_time)


def hill_climbing(problem, iterations_limit=0, viewer=None):
    '''
    Hill climbing search.

    If iterations_limit is specified, the algorithm will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.actions, SearchProblem.result, and
    SearchProblem.value.
    '''
    return _local_search(problem,
                         _first_expander,
                         iterations_limit=iterations_limit,
                         fringe_size=1,
                         stop_when_no_better=True,
                         viewer=viewer)


def _random_best_expander(fringe, iteration, viewer, problem):
    '''
    Expander that expands one randomly chosen nodes on the fringe that
    is better than the current (first) node.
    '''
    current = fringe[0]
    neighbors = current.expand(local_search=True)
    if viewer:
        viewer.event('expanded', [current], [neighbors])

    betters = [n for n in neighbors
               if n.value > current.value]
    if betters:
        chosen = random.choice(betters)
        if viewer:
            viewer.event('chosen_node', chosen)
        fringe.append(chosen)


def hill_climbing_stochastic(problem, iterations_limit=0, viewer=None, max_run_time=1, seed=0):
    '''
    Stochastic hill climbing.

    If iterations_limit is specified, the algorithm will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.actions, SearchProblem.result, and
    SearchProblem.value.
    '''
    random.seed(seed)
    return _local_search(problem,
                         _random_best_expander,
                         iterations_limit=iterations_limit,
                         fringe_size=1,
                         stop_when_no_better=iterations_limit==0,
                         viewer=viewer,
                         max_run_time=max_run_time)


def _greedy_expander(cur_sol, iteration, cur_iter, problem):
    '''
    Expander that expands the best greedy choice for cur_iter.

    Notes:
        using cur_iter here instead of iteration because if its GREEDYLOOP the cur_iter needs to reset.
        also used for GREEDYLOOP via use of a flag.
    '''
    current_sol_vec = deepcopy(cur_sol)
    best_sol_vec = SolutionVector()
    best_grade_vec = GradesVector().scalarize()
    flag = False

    if cur_iter >= problem.valuesPerVariables.validVarAmount:
        if problem.algoName == "GREEDYLOOP":
            current_sol_vec = SolutionVector()
            problem.valuesPerVariables.varsData.sort(key=lambda x: x.ucPrio)
            cur_iter = 0
            flag = True
        else:
            return 'STOP_GREEDY'

    for curVarValue in range(problem.maxValuesNum):
        current_sol_vec.solutionVector[cur_iter] = curVarValue
        cur_grade_vec = problem.value(current_sol_vec)
        if cur_grade_vec > best_grade_vec:
            best_sol_vec.solutionVector = deepcopy(current_sol_vec.solutionVector)
            best_grade_vec = cur_grade_vec

    if flag:
        flag = False
        return best_sol_vec, -1

    return best_sol_vec


def greedy(problem, iterations_limit=0, viewer=None, max_run_time=1, seed=0):
    '''
    greedy.

    If iterations_limit is specified, the algorithm will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.actions, SearchProblem.result, and
    SearchProblem.value.
    '''
    problem.valuesPerVariables.varsData.sort(key=lambda x: x.ucPrio)
    random.seed(seed)

    return _local_search(problem,
                         _greedy_expander,
                         iterations_limit=iterations_limit,
                         fringe_size=1,
                         stop_when_no_better=iterations_limit==0,
                         viewer=viewer,
                         max_run_time=max_run_time)


def _random_search_expander(cur_sol, iteration, viewer, problem):
    '''
    Expander that returns a random expansion not based on anything prior.
    '''
    current_sol_vec = deepcopy(cur_sol)

    for curVarValue in range(MAX_NUM_OF_VARS):
        rand_chosen = random.randint(0, problem.valuesPerVariables.varsData[curVarValue].valuesAmount - 1)
        current_sol_vec.solutionVector[curVarValue] = rand_chosen

    return current_sol_vec


def random_search(problem, iterations_limit=0, viewer=None, max_run_time=1, seed=0):
    '''
    random search.

    If iterations_limit is specified, the algorithm will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.actions, SearchProblem.result, and
    SearchProblem.value.
    '''
    random.seed(seed)

    return _local_search(problem,
                         _random_search_expander,
                         iterations_limit=iterations_limit,
                         fringe_size=1,
                         stop_when_no_better=iterations_limit==0,
                         viewer=viewer,
                         max_run_time=max_run_time)


def hill_climbing_random_restarts(problem, restarts_limit, iterations_limit=0, viewer=None):
    '''
    Hill climbing with random restarts.

    restarts_limit specifies the number of times hill_climbing will be runned.
    If iterations_limit is specified, each hill_climbing will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.actions, SearchProblem.result, SearchProblem.value,
    and SearchProblem.generate_random_state.
    '''
    restarts = 0
    best = None

    while restarts < restarts_limit:
        new = _local_search(problem,
                            _first_expander,
                            iterations_limit=iterations_limit,
                            fringe_size=1,
                            random_initial_states=True,
                            stop_when_no_better=True,
                            viewer=viewer)

        if not best or best.value < new.value:
            best = new

        restarts += 1

    if viewer:
        viewer.event('no_more_runs', best, 'returned after %i runs' % restarts_limit)

    return best


# Math literally copied from aima-python
def _exp_schedule(iteration, k=20, lam=0.005, limit=100):
    '''
    Possible scheduler for simulated_annealing, based on the aima example.
    '''
    return k * math.exp(-lam * iteration)


def _create_simulated_annealing_expander(schedule, seed, initTemp, tempStep):
    '''
    Creates an expander that has a random chance to choose a node that is worse
    than the current (first) node, but that chance decreases with time.
    '''
    # random.seed(seed)

    def _expander(fringe, iteration, viewer, problem):
        T = schedule(iteration, k=initTemp, lam=tempStep)
        current = fringe[0]
        neighbors = current.expand(local_search=True)

        if viewer:
            viewer.event('expanded', [current], [neighbors])

        if neighbors:
            succ = random.choice(neighbors)
            delta_e = succ.value - current.value

            if delta_e > 0 or random.random() < np.float64(math.exp(np.float64(-abs(delta_e / T)))):
                fringe.pop()
                fringe.append(succ)

                if viewer:
                    viewer.event('chosen_node', succ)

    return _expander


# schedule=_exp_schedule
def simulated_annealing(problem, schedule=_exp_schedule,
                        iterations_limit=0, viewer=None,
                        max_run_time=1, seed=0,
                        initTemp=10, tempStep=1):
    '''
    Simulated annealing.

    schedule is the scheduling function that decides the chance to choose worst
    nodes depending on the time.
    If iterations_limit is specified, the algorithm will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.actions, SearchProblem.result, and
    SearchProblem.value.
    '''
    random.seed(seed)
    return _local_search(problem,
                         _create_simulated_annealing_expander(schedule, seed, initTemp, tempStep),
                         iterations_limit=iterations_limit,
                         fringe_size=1,
                         stop_when_no_better=iterations_limit==0,
                         viewer=viewer,
                         max_run_time=max_run_time)


def _create_genetic_expander(problem, mutation_chance):
    '''
    Creates an expander that expands the bests nodes of the population,
    crossing over them.
    '''
    def _expander(fringe, iteration, viewer):
        fitness = [x.value for x in fringe]
        sampler = InverseTransformSampler(fitness, fringe)
        new_generation = []

        expanded_nodes = []
        expanded_neighbors = []

        for _ in fringe:
            node1 = sampler.sample()
            node2 = sampler.sample()
            child = problem.crossover(node1.state, node2.state)
            action = 'crossover'
            if random.random() < mutation_chance:
                # Noooouuu! she is... he is... *IT* is a mutant!
                child = problem.mutate(child)
                action += '+mutation'

            child_node = SearchNodeValueOrdered(state=child, problem=problem, action=action)
            new_generation.append(child_node)

            expanded_nodes.append(node1)
            expanded_neighbors.append([child_node])
            expanded_nodes.append(node2)
            expanded_neighbors.append([child_node])

        if viewer:
            viewer.event('expanded', expanded_nodes, expanded_neighbors)

        fringe.clear()
        for node in new_generation:
            fringe.append(node)

    return _expander


def genetic(problem, population_size=100, mutation_chance=0.1,
            iterations_limit=0, viewer=None):
    '''
    Genetic search.

    population_size specifies the size of the population (ORLY).
    mutation_chance specifies the probability of a mutation on a child,
    varying from 0 to 1.
    If iterations_limit is specified, the algorithm will end after that
    number of iterations. Else, it will continue until it can't find a
    better node than the current one.
    Requires: SearchProblem.generate_random_state, SearchProblem.crossover,
    SearchProblem.mutate and SearchProblem.value.
    '''
    return _local_search(problem,
                         _create_genetic_expander(problem, mutation_chance),
                         iterations_limit=iterations_limit,
                         fringe_size=population_size,
                         random_initial_states=True,
                         stop_when_no_better=iterations_limit==0,
                         viewer=viewer)


def writeResults(path, time_array, best_val_array, problem):
    """Writing results of algorithms into appropriate files in path.

    Args:
        path (str): path to where the results will be written into.
        time_array (list[timedelta]): array of times at specific iteration
        best_val_array (list[(SolutionVector, float)]): array of tuples, first is SolutionVector, second is its scalarized evaluation.
        problem (class): the problem that we defined, in out case its COP

    Returns:
         None.
    """
    min_time = min(time_array)
    if min_time < 0:
        array_of_times = list(map(lambda x: x + abs(min_time), time_array))[::-1]
    else:
        array_of_times = list(map(lambda x: x - abs(min_time), time_array))[::-1]

    best_val_array = sorted(best_val_array, key=lambda x: x[1])

    with open(path + "\Times_" + str(problem.algoName) + "_Problem_" + str(problem.problemSeed) + "_RunSeed_" + str(
            problem.algoSeed) + ".txt", 'w') as f:
        for line in array_of_times:
            f.write(str(line) + '\n')

    with open(path + "\BestValue_" + str(problem.algoName) + "_Problem_" + str(problem.problemSeed) + "_RunSeed_" + str(
            problem.algoSeed) + ".txt", 'w') as f:
        for vec, value in best_val_array:
            f.write(str(value) + '\n')


def _local_search(problem, fringe_expander, iterations_limit=0, fringe_size=1,
                  random_initial_states=False, stop_when_no_better=True,
                  viewer=None, max_run_time=1):
    '''
    Basic algorithm for all local search algorithms.
    '''
    if viewer:
        viewer.event('started')

    fringe = BoundedPriorityQueue(fringe_size)
    curr_sol = None
    best = None
    if random_initial_states:
        for _ in range(fringe_size):
            s = problem.generate_random_state()
            fringe.append(SearchNodeValueOrdered(state=s, problem=problem))
    else:
        if 'InitialGreedy' in problem.algoName or problem.algoName in ['GREEDY', 'GREEDYLOOP', 'RS', 'GREEDY+RS']:
            curr_sol = problem.initial_state
            best = curr_sol
        elif problem.algoName in ['RW', 'GREEDY+RW']:
            curr_sol = SearchNodeValueOrdered(state=problem.initial_state,
                                                 problem=problem)
            best = curr_sol
        else:
            fringe.append(SearchNodeValueOrdered(state=problem.initial_state,
                                                 problem=problem))
            best = fringe[0]

    finish_reason = ''
    iteration = 0
    cur_iter = 0
    run = True
    curr = None
    stop = datetime.datetime.now() + datetime.timedelta(seconds=max_run_time)

    array_of_best_solutions_values = []
    array_of_times = []
    while run and datetime.datetime.now() < stop:
        if viewer:
            viewer.event('new_iteration', list(fringe))

        if 'InitialGreedy' in problem.algoName or problem.algoName in ['GREEDY', 'GREEDYLOOP', 'RS', 'GREEDY+RS']:
            old_best = curr_sol
            curr = fringe_expander(curr_sol, iteration, cur_iter, problem)

            if type(curr) is tuple:  # to handle greedyloop
                curr, cur_iter = curr

            if curr != 'STOP_GREEDY' and problem.value(best) < problem.value(curr):
                best = curr

            cur_iter += 1


            curr_sol = curr
        else:
            if problem.algoName in ['RW', 'GREEDY+RW']:
                old_best = curr_sol
                curr = fringe_expander(curr_sol, iteration, viewer, problem)[0]

                if curr != 'STOP_GREEDY' and best.value < curr.value:
                    best = curr

                curr_sol = curr
            else:
                old_best = fringe[0]
                fringe_expander(fringe, iteration, viewer, problem)
                best = fringe[0]

        if curr == 'STOP_GREEDY':
            break

        if 'InitialGreedy' in problem.algoName or problem.algoName in ['GREEDY', 'GREEDYLOOP', 'RS', 'GREEDY+RS']:
            array_of_best_solutions_values.append((best, problem.value(best)))
        else:
            array_of_best_solutions_values.append((best, best.value))
        array_of_times.append((stop - datetime.datetime.now()).total_seconds())
        iteration += 1

        if iterations_limit and iteration >= iterations_limit:
            run = False
            finish_reason = 'reaching iteration limit'
        elif ('InitialGreedy' in problem.algoName or problem.algoName in ['GREEDY', 'GREEDYLOOP', 'RS', 'GREEDY+RS']) and old_best >= best and stop_when_no_better:
            run = False
            finish_reason = 'greedy stopped, not being able to improve solution'
        elif 'InitialGreedy' not in problem.algoName and problem.algoName not in ['GREEDY', 'GREEDYLOOP', 'RS', 'GREEDY+RS'] and old_best.value >= best.value and stop_when_no_better:
            run = False
            finish_reason = 'not being able to improve solution'

    if viewer:
        viewer.event('finished', fringe, best, 'returned after %s' % finish_reason)

    #########################################################################################
    ############# writing info to files #####################################################
    #########################################################################################

    path = os.path.join(os.getcwd(), '..', 'copsimpleai', 'pythonLocalSearch', 'Results')
    writeResults(path, array_of_times, array_of_best_solutions_values, problem)


    # return best, original implementation didn't return the best, it returned the last iteration.
    # flag = False
    # if problem.algoName in ['RS', 'GREEDY+RS', 'RW', 'GREEDY+RW']:
    #     if 'InitialGreedy' in problem.algoName:
    #         flag = True
    #     if not flag:
    #         return best

    best_res = max(array_of_best_solutions_values, key=lambda x: x[1])
    return best_res[0]