class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def Max(state,d,alpha,beta):
            if state.isWin() or state.isLose() or d==self.depth:
                return self.evaluationFunction(state)
            temp=float("-inf")
            for action in state.getLegalActions(0):
                temp=max(temp,Min(state.generateSuccessor(0,action),d,1,alpha,beta))
                if temp>beta:
                    return temp
                alpha=max(alpha,temp)
            return temp
        def Min(state,d,agent_ind,alpha,beta):
            if state.isWin() or state.isLose() or d==self.depth:
                return self.evaluationFunction(state)
            temp=float("inf")
            for action in state.getLegalActions(agent_ind):
                if agent_ind==state.getNumAgents()-1:
                    temp=min(temp,Max(state.generateSuccessor(agent_ind,action),d+1,alpha,beta))
                else:
                    temp=min(temp,Min(state.generateSuccessor(agent_ind,action),d,agent_ind+1,alpha,beta))
                if temp<alpha:
                    return temp
                beta=min(beta,temp)
            return temp
        val=float("-inf")
        alpha=float("-inf")
        beta=float("inf")
        for action in gameState.getLegalActions(0):
            temp = Min(gameState.generateSuccessor(0,action), 0, 1,alpha,beta)
            if temp > val:
                val = temp
                move = action
                alpha=max(temp,alpha)
        return move