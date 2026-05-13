
# check every parameter. 

import numpy as np
from matplotlib import pyplot as plt
from utilities import CellType
from main import run_agent
from itertools import product
from map_abstraction import MapAbstraction


"""
what to check for:
1. sarsa vs. q-learn with same params. (check for best alpha here?)
2. compare epsilons using gamma = 0.5

"""

def run_suite():
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    epsilons = [0, 0.5, 1]
    algs = ["sarsa", "qlearn"]

    # stage 1: compare sarsa and q-learn on all maps. 

    # combos = [(w, 0.5, 0.5, z) for w in alphas for z in algs]
    # for i in range (1, 5):  # for each map:
    #     print("Map ", i)
    #     for alpha, gamma, epsilon, alg in combos: # for each combo
    #         print("    alpha: ", alpha, ", gamma: ", gamma, ", epsilon: ", epsilon, ", alg: ", alg)
    #         run_agent(alpha, gamma, epsilon, alg, map_num=i)

    # arbitrary value until further tested. 
    # sike, I actually do not know what the best alpha is.

    # stage 2: compare epsilons on map 4 using sarsa and qlearn

    # seems that an alpha of 0.3 was best for SARSA on map 4 during initial exploration. 
    # s2_combos = [(0.5, 0.5, x) for x in epsilons]
    # for alpha, gamma, epsilon in s2_combos:
    #     print("    alpha: ", alpha, ", gamma: ", gamma, ", epsilon: ", epsilon)
    #     run_agent(alpha, gamma, epsilon, "qlearn", 4)

    # stage 3: compare gammas on map 4 using sarsa and qlearn

    print("-----  STAGE 3  -----")

    s3_gammas = [0.1, 0.5, 1]
    s3_combos = [(0.5, x, 0.5, y) for x in s3_gammas for y in algs]
    for alpha, gamma, epsilon, alg in s3_combos:
        print("    alpha: ", alpha, ", gamma: ", gamma, ", epsilon: ", epsilon)
        run_agent(alpha, gamma, epsilon, alg, 4)

    # stage 4: best combos
    # epsilon 0.5 and gamma 1 seemed best. fixing alpha at 0.5

    # NOTE: SPECIFY STRATEGY in main, this was not added as a tunable parameter. 
    # strategy = "S2"
    # for alg in algs: 
    #     print("strategy: ", strategy, ", alg: ", alg)
    #     run_agent(0.5, 1, 0.5, alg=alg, map_num=4)
    




def main():
    run_suite()

if __name__ == "__main__":
    main()

