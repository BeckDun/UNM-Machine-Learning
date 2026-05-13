from environment import Environment
from agent import Agent
from strategy import NaiveStrategy, DistanceShapedStrategy
import q_learning
import sarsa
import test_agent
from typing import Literal

def run_agent(alpha : float, gamma : float, epsilon : float, alg : Literal["sarsa, qlearn"] = "sarsa", 
              map_num : Literal[1, 2, 3, 4] = 1):
    """
    A utility method to run the agent and make any necessary function calls.
    Runs training and evaluates the testing accuracy of the agent at the end. 

    Params
    _____
    alpha: a float representing the learning rate to use. 

    gamma: a float representing the discount value. 

    epsilon: a float representing the exploration rate. 

    alg: a string representing which learning algorithm to use; can be either SARSA or Q-learning. 

    map_num: an integer representing which of the four maps to use. 
    
    """
    bmp_str = "maps/map" + str(map_num) + ".bmp"
    enviroment = Environment(bmp_file=bmp_str)
    # agent = Agent(enviroment, alpha = 0.6, gamma = 0.5, epsilon = 0.0)
    agent = Agent(enviroment, alpha, gamma, epsilon)
    # strategy = NaiveStrategy()
    strategy = DistanceShapedStrategy(enviroment.target)

    # Run training and testing on the specified algorithm. 
    if alg == "qlearn":
        print("Training with Q-learning")
        q_learning.traning_loop(enviroment, agent, strategy, episodes=5000,max_steps=2000)

        print("Testing Q-learning")
        test_agent.test_accuracy(enviroment, agent, strategy, max_steps=200)

    elif alg == "sarsa":
        print("Training with SARSA")
        sarsa.traning_loop(enviroment, agent, strategy, episodes=5000, max_steps=2000)

        print("Testing SARSA")
        test_agent.test_accuracy(enviroment, agent, strategy, max_steps=200)


if __name__ == "__main__":
    run_agent()
