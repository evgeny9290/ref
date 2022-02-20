# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghostPos = []
        for ghost in newGhostStates:
            ghostPos.append((ghost.getPosition()[0], ghost.getPosition()[1]))

        if newScaredTimes[0] <= 0 and (newPos in ghostPos):
            return -1

        if newPos in currentGameState.getFood().asList():
            return 1

        closest_food_dist = float('inf')
        newFood = newFood.asList()
        for food_pos in newFood:
            closest_food_dist = min(closest_food_dist, util.manhattanDistance(newPos, food_pos))

        closest_ghost_dist = float('inf')
        for g_loc in ghostPos:
            closest_ghost_dist = min(closest_ghost_dist, util.manhattanDistance(newPos, g_loc))

        return 1 / closest_food_dist - 1 / closest_ghost_dist
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        ##############################################################################
        ######### implementation number one ##########################################
        ##############################################################################

        # def stop(state, depth):
        #     return state.isWin() or state.isLose() or depth == self.depth
        #
        # def val(state, depth, agent_idx):
        #     if stop(state, depth):
        #         return self.evaluationFunction(state)
        #     if agent_idx == 0:
        #         return max_val(state, depth)
        #     if agent_idx > 0:
        #         return min_val(state, depth, agent_idx)
        #
        # def max_val(state, depth):
        #     v = float('-inf')
        #     act = ''
        #     for action in state.getLegalActions(0):
        #         v_cur = val(state.generateSuccessor(0, action), depth, 1)
        #         v_cur = v_cur[0] if type(v_cur) == list else v_cur
        #         if v < v_cur:
        #             v = v_cur
        #             act = action
        #     return [v, act]
        #
        # def min_val(state, depth, agent_idx):
        #     v = float('inf')
        #     act = ''
        #     for action in state.getLegalActions(agent_idx):
        #         if agent_idx == state.getNumAgents() - 1:
        #             v_cur = val(state.generateSuccessor(agent_idx, action), depth + 1, 0)
        #         else:
        #             v_cur = val(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1)
        #         v_cur = v_cur[0] if type(v_cur) == list else v_cur
        #         if v > v_cur:
        #             v = v_cur
        #             act = action
        #     return [v, act]
        #
        # return val(gameState, 0, 0)[1]

        ##############################################################################
        ######### implementation number two ##########################################
        ##############################################################################

        def stop(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def max_val(state, depth):
            if stop(state, depth):
                return self.evaluationFunction(state)
            v = float('-inf')
            # next_agent_states = [state.generateSuccessor(0, action) for action in state.getLegalActions(0)]
            for action in state.getLegalActions(0):
                v = max(v, min_val(state.generateSuccessor(0, action), depth, 1))
            return v

        def min_val(state, depth, ghost):
            if stop(state, depth):
                return self.evaluationFunction(state)
            v = float('inf')
            # next_ghost_states = [state.generateSuccessor(ghost, action) for action in state.getLegalActions(ghost)]
            for action in state.getLegalActions(ghost):
                if ghost == state.getNumAgents() - 1:
                    v = min(v, max_val(state.generateSuccessor(ghost, action), depth + 1))
                else:
                    v = min(v, min_val(state.generateSuccessor(ghost, action), depth, ghost + 1))
            return v

        v = float('-inf')
        act = gameState.getLegalActions(0)[0]
        for action in gameState.getLegalActions(0):
            v_curr = min_val(gameState.generateSuccessor(0, action), 0, 1)
            if v_curr > v:
                v = v_curr
                act = action

        return act
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def stop(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def max_val(state, alpha, beta, depth):
            if stop(state, depth):
                return self.evaluationFunction(state)
            v = float('-inf')
            ##################################################################################################
            # this approach is correct but their auto grader checks for expanded nodes and this way we expand all nodes
            # and after expansion we do alpha beta pruning but this is defeates the purpose of pruning
            # if you expand first, hence more nodes expansion than actually needed
            ##################################################################################################
            # next_agent_states = [state.generateSuccessor(0, action) for action in state.getLegalActions(0)]
            # for s in next_agent_states:
            #     v = max(v, min_val(s, alpha, beta, depth, 1))
            #     if v > beta:
            #         return v
            #     alpha = max(alpha, v)
            ###################################################################################################
            for action in state.getLegalActions(0):
                v = max(v, min_val(state.generateSuccessor(0, action), alpha, beta, depth, 1))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_val(state, alpha, beta, depth, ghost):
            if stop(state, depth):
                return self.evaluationFunction(state)
            v = float('inf')
            ##################################################################################################
            # this approach is correct but their auto grader checks for expanded nodes and this way we expand all nodes
            # and after expansion we do alpha beta pruning but this is defeates the purpose of pruning
            # if you expand first, hence more nodes expansion than actually needed
            ##################################################################################################
            # next_ghost_states = [state.generateSuccessor(ghost, action) for action in state.getLegalActions(ghost)]
            # for s in next_ghost_states:
            #     if ghost == state.getNumAgents() - 1:
            #         v = min(v, max_val(s, alpha, beta, depth + 1))
            #     else:
            #         v = min(v, min_val(s, alpha, beta, depth, ghost + 1))
            #     if v < alpha:
            #         return v
            #     beta = min(beta, v)
            ###################################################################################################
            for action in state.getLegalActions(ghost):
                if ghost == state.getNumAgents() - 1:
                    v = min(v, max_val(state.generateSuccessor(ghost, action), alpha, beta, depth + 1))
                else:
                    v = min(v, min_val(state.generateSuccessor(ghost, action), alpha, beta, depth, ghost + 1))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        v = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        act = gameState.getLegalActions()[0]
        for action in gameState.getLegalActions(0):
            v_curr = min_val(gameState.generateSuccessor(0, action), alpha, beta, 0, 1)
            if v_curr > v:
                v = v_curr
                act = action
                alpha = max(v_curr, alpha)

        return act
        # util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def stop(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def mav_val(state, depth):
            if stop(state, depth):
                return self.evaluationFunction(state)
            v = float('-inf')
            for action in state.getLegalActions(0):
                v = max(v, exp_val(state.generateSuccessor(0, action), depth, 1))
            return v

        def exp_val(state, depth, ghost):
            if stop(state, depth):
                return self.evaluationFunction(state)
            v = 0
            for action in state.getLegalActions(ghost):
                if ghost == state.getNumAgents() - 1:
                    v += mav_val(state.generateSuccessor(ghost, action), depth + 1)
                else:
                    v += exp_val(state.generateSuccessor(ghost, action), depth, ghost + 1)
            return v / len(state.getLegalActions(ghost))

        v = float('-inf')
        act = gameState.getLegalActions()[0]
        for action in gameState.getLegalActions(0):
            v_curr = exp_val(gameState.generateSuccessor(0, action), 0, 1)
            if v_curr > v:
                v = v_curr
                act = action

        return act
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    capsuleLocations = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = 0
    ghostPos = []
    for ghost in newGhostStates:
        ghostPos.append((ghost.getPosition()[0], ghost.getPosition()[1]))

    if newScaredTimes[0] <= 0 and (newPos in ghostPos):
        return -1

    if newPos in currentGameState.getFood().asList():
        return 1

    closest_food_dist = float('inf')
    newFood = newFood.asList()
    for food_pos in newFood:
        closest_food_dist = min(closest_food_dist, util.manhattanDistance(newPos, food_pos))

    closest_ghost_dist = float('inf')
    for g_loc in ghostPos:
        closest_ghost_dist = min(closest_ghost_dist, util.manhattanDistance(newPos, g_loc))

    score += 1 / closest_food_dist + 1 / closest_ghost_dist + currentGameState.getScore()
    return score

    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
