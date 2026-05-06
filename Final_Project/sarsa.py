def traning_loop(enviroment, agent, strategy, episodes = 1000):
    for _ in range(episodes):

        state = enviroment.agent_start
        action = agent.choose_action(state)

        done = False
        while not done:
            next_state, next_action, reward, done = agent.step(state, action, strategy, method="sarsa")
            state = next_state
            
            if not done:
                action = next_action