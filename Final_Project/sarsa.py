from matplotlib import pyplot as plt

def traning_loop(environment, agent, strategy, episodes = 1000, max_steps = 500):
    for episode in range(episodes):
        print(f"Episode {episode}")
        steps = 0
        state = environment.agent_start
        action = agent.choose_action(state)
        done = False

        while not done and steps < max_steps:
            next_state, next_action, _, done = agent.step(state, action, strategy, method="sarsa")
            steps += 1
            state = next_state
            
            if not done:
                action = next_action
                
    