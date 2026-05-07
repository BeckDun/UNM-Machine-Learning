from matplotlib import pyplot as plt

def traning_loop(environment, agent, strategy, episodes=1000, max_steps = 500):
    #plt.figure(figsize=(8, 8))
    
    for episode in range(episodes):
        steps = 0
        print(f"Episode {episode}")
        state = environment.agent_start
        done = False

        while not done and steps < max_steps:
            #if episode % 100 == 0:
                #environment.plot_environment(environment.map, state)

            action = agent.choose_action(state)
            next_state, _, _, done = agent.step(state, action, strategy, method="q_learning")
            steps += 1

            state = next_state
            if done:
                print(f"Reached target in episode {episode}")
        print(f"Total Steps {steps}")

    #environment.plot_environment(environment.map, state)
    #plt.show()
