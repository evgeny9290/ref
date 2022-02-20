# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
from collections import defaultdict
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            val_dict = self.values.copy()
            for s in self.mdp.getStates():
                val_max = float('-inf')
                valid_actions = self.mdp.getPossibleActions(s)
                for action in valid_actions:
                    q_val = self.computeQValueFromValues(s, action)
                    val_max = max(val_max, q_val)
                    val_dict[s] = val_max
            self.values = val_dict


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        q_val = 0
        for s_prime, prob in transitions:
            q_val += prob * (self.mdp.getReward(state, action, s_prime) + self.discount * self.values[s_prime])
        return q_val
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        best_action = None
        v_max = float('-inf')
        for action in actions:
           v = self.computeQValueFromValues(state, action)
           if v > v_max:
                v_max = v
                best_action = action
        return best_action

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        num_states = len(states)
        for i in range(self.iterations):
            state = states[i % num_states]
            if not self.mdp.isTerminal(state):
                val_max = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    q_val = self.computeQValueFromValues(state, action)
                    val_max = max(val_max, q_val)
                self.values[state] = val_max


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        pred_dict = defaultdict(set)
        not_terminal_states = [state for state in states if not self.mdp.isTerminal(state)]

        # calculating pred_dict i.e predecessors of a given state
        for state in not_terminal_states:
            for action in self.mdp.getPossibleActions(state):
                next_states = [nxt_state[0] for nxt_state in self.mdp.getTransitionStatesAndProbs(state, action)]
                for nxt_state in next_states:
                    pred_dict[nxt_state].add(state)

        prio_que = util.PriorityQueue()
        # filling the prio que (max heap) hence -diff
        for state in not_terminal_states:
            if not self.mdp.isTerminal(state):
                state_val = self.values[state]
                q_val_max = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    q_val_max = max(q_val_max, self.computeQValueFromValues(state, action))
                diff = abs(q_val_max - state_val)
                prio_que.update(state, -diff)

        for i in range(self.iterations):
            if prio_que.isEmpty():
                break
            state = prio_que.pop()
            if not self.mdp.isTerminal(state):
                max_q = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    max_q = max(max_q, self.computeQValueFromValues(state, action))
                self.values[state] = max_q
                for pred in pred_dict[state]:
                    pred_val = self.values[pred]
                    q_val_max = float('-inf')
                    for action in self.mdp.getPossibleActions(pred):
                        q_val_max = max(q_val_max, self.computeQValueFromValues(pred, action))
                    diff = abs(q_val_max - pred_val)
                    if diff > self.theta:
                        prio_que.update(pred, -diff)





