from matplotlib import pyplot as plt

def traning_loop(environment, agent, strategy, episodes = 1000):

    for episode in range(episodes):

        state = environment.agent_start
        action = agent.choose_action(state)

        done = False

        plt.figure(figsize=(8, 8))
        while not done:
            if episode % 100 == 0:
                environment.plot_environment(environment.map, state)
    
            next_state, next_action, reward, done = agent.step(state, action, strategy, method="sarsa")
            state = next_state
            
            if not done:
                action = next_action
                
    environment.plot_environment(environment.map, state)
    plt.show()