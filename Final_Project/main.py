from environment import Environment
from agent import Agent
from strategy import NaiveStrategy
import q_learning
import sarsa
import test_agent

def main():
    enviroment = Environment(bmp_file="maps/map4.bmp")
    agent = Agent(enviroment, alpha = 0.1, gamma = 0.9, epsilon = 0.1)
    strategy = NaiveStrategy()

    #print("Training with Q-learning")
    #q_learning.traning_loop(enviroment, agent, strategy, episodes=5000,max_steps=2000)

    print("Training with SARSA")
    sarsa.traning_loop(enviroment, agent, strategy, episodes=5000, max_steps=2000)

    print("Testing learned policy")
    test_agent.test_agent(enviroment, agent, strategy, max_steps=200)


if __name__ == "__main__":
    main()
