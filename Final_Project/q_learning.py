def traning_loop(enviroment, agent, strategy, episodes=1000):
    for _ in range(episodes):
        state = enviroment.agent_start
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, next_action, reward, done = agent.step(state, action, strategy, method="q_learning")

            state = next_state
