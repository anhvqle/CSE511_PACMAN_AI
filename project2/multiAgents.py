# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    some Directions.X for some X in the set {North, South, West, East, Stop}
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

    if action == Directions.STOP:
        return -float('inf')

    if successorGameState.isWin():
        return float('inf')

    closestFood = float('inf')
    foodList = newFood.asList()
    if len(foodList) == 0:
        return float('inf')
    for food in foodList:
        closestFood = min(closestFood, manhattanDistance(newPos, food))

    closetGhost = float('inf')
    ghostList = [ghost.getPosition() for ghost in newGhostStates]
    
    for ghost in ghostList:
        closetGhost = min(closetGhost, manhattanDistance(newPos, ghost))
    
    score = successorGameState.getScore() + closetGhost - closestFood
    
    return score

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

        Directions.STOP:
            The stop direction, which is always legal

        gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        return self.max_value(gameState, self.depth)[1]

    def max_value(self, gameState, depth):
        legalMoves = gameState.getLegalActions(0)

        if depth == 0 or not legalMoves:
            return self.evaluationFunction(gameState), Directions.STOP
        
        v_total = -float('inf')
        a_total = Directions.STOP

        for move in legalMoves:
            if move == Directions.STOP:
                continue
            v,a = self.min_value(gameState.generateSuccessor(0, move), 1, depth - 1)
            if v_total < v:
                v_total = v
                a_total = move

        return v_total, a_total

    def min_value(self, gameState, agentIndex, depth):

        legalMoves = gameState.getLegalActions(agentIndex)
        if depth == 0 or not legalMoves:
            return self.evaluationFunction(gameState), Directions.STOP
            
        numAgents = gameState.getNumAgents()

        v_total = float('inf')
        a_total = Directions.STOP

        for move in legalMoves:
            if move == Directions.STOP:
                continue
            state = gameState.generateSuccessor(agentIndex, move)
            
            if agentIndex == numAgents - 1:
                v,a = self.max_value(state, depth - 1)
            else:
                v,a = self.min_value(state, agentIndex + 1, depth)

            if v_total > v:
                v_total = v
                a_total = move

        return v_total, a_total








class AlphaBetaAgent(MultiAgentSearchAgent):
    """
        Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        return self.max_value(gameState, self.depth, -float('inf'), float('inf'))[1]

    def max_value(self, gameState, depth, alpha, beta):
        legalMoves = gameState.getLegalActions(0)

        if depth == 0 or not legalMoves:
            return self.evaluationFunction(gameState), Directions.STOP
        
        v_total = -float('inf')
        a_total = Directions.STOP

        for move in legalMoves:
            if move == Directions.STOP:
                continue
            v,a = self.min_value(gameState.generateSuccessor(0, move), 1, depth - 1, alpha, beta)
            
            if v_total < v:
                v_total = v
                a_total = move
            
            alpha = max(alpha, v_total)
            if alpha >= beta:
                break

        return v_total, a_total

    def min_value(self, gameState, agentIndex, depth, alpha, beta):

        legalMoves = gameState.getLegalActions(agentIndex)
        if depth == 0 or not legalMoves:
            return self.evaluationFunction(gameState), Directions.STOP
            
        numAgents = gameState.getNumAgents()

        v_total = float('inf')
        a_total = Directions.STOP

        for move in legalMoves:
            if move == Directions.STOP:
                continue
            state = gameState.generateSuccessor(agentIndex, move)
            
            if agentIndex == numAgents - 1:
                v,a = self.max_value(state, depth - 1, alpha, beta)
            else:
                v,a = self.min_value(state, agentIndex + 1, depth, alpha, beta)

            if v_total > v:
                v_total = v
                a_total = move
            
            beta = min(beta, v_total)
            if alpha >= beta:
                break

        return v_total, a_total




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

        # return self.max_value(gameState, self.depth)[1]
        v_total = -float('inf')
        nextMove = Directions.STOP
        legalMoves = gameState.getLegalActions(0)
        for move in legalMoves:
            v = self.exp_value(gameState.generateSuccessor(0, move), 1, self.depth)[0]
            if v > v_total and move != Directions.STOP:
                v_total = v
                nextMove = move

        return nextMove


    def max_value(self, gameState, depth):
        if depth == 0:
            return self.evaluationFunction(gameState), Directions.STOP

        legalMoves = gameState.getLegalActions(0)
        a_total = Directions.STOP

        if len(legalMoves) <= 0:
            v_total = self.evaluationFunction(gameState)
        else:
            v_total = -float('inf')

        for move in legalMoves:
            nextState = gameState.generateSuccessor(0, move)
            v = self.exp_value(nextState, 1, depth)[0]
            if v_total < v:
                v_total = v
                a_total = move

        return v_total, a_total

    def exp_value(self, gameState, agentIndex, depth):
        if depth == 0:
            return self.evaluationFunction(gameState), Directions.STOP

        v_total = 0;
        legalMoves = gameState.getLegalActions(agentIndex)
        a_total = Directions.STOP
        
        if len(legalMoves) == 0:
            return self.evaluationFunction(gameState), Directions.STOP

        p = 1.0 / len(legalMoves)
        for move in legalMoves:
            a_total = move
            nextState = gameState.generateSuccessor(agentIndex, move)
            
            if agentIndex == gameState.getNumAgents() - 1:
                v, a = self.max_value(nextState, depth-1)

            else:
                v, a = self.exp_value(nextState, agentIndex+1, depth)
                
            v_total += p * v

        return v_total, a_total






def betterEvaluationFunction(currentGameState):
    """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (question 5).

        DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    currPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    foodList = foodGrid.asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    legalAction = currentGameState.getLegalActions()
    if len(foodList) == 0:
        return float('inf')
    closestFood = float('inf')
    for food in foodList:
        closestFood = min(closestFood, manhattanDistance(currPos, food))

    closetGhost = float('inf')
    ghostList = [ghost.getPosition() for ghost in newGhostStates]
    for ghost in ghostList:
        closetGhost = min(closetGhost, manhattanDistance(currPos, ghost))

    return currentGameState.getScore() + closetGhost * 5 - closestFood - len(foodList) * 10

# Abbreviation
better = betterEvaluationFunction






class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR C.ODE HERE ***"
    util.raiseNotDefined()

