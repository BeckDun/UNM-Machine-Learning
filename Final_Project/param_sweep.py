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
    gammas = [0.1, 0.3, 0.5, 0.7, 0.9]
    epsilons = [0, 0.5, 1]
    algs = ["sarsa", "qlearn"]

    # stage 1: compare sarsa and q-learn on all maps. 
    combos = [(w, 0.5, 0.5, z) for w in alphas for z in algs]
    for i in range (1, 5):  # for each map:
        ("Map ", i)
        for alpha, gamma, epsilon, alg in combos: # for each combo
            print("    alpha: ", alpha, ", gamma: ", gamma, ", epsilon: ", epsilon, ", alg: ", alg)
            run_agent(alpha, gamma, epsilon, alg, map_num=i)

    # arbitrary value until further tested. 
    best_alpha = 0.5

    epsilon_zero = [(x, 0.5, 0.0) for x in alphas for y in gammas]
    epsilon_half = [(x, 0.5, 0.5) for x in alphas for y in gammas]
    epsilon_one = [(x, 0.5, 1) for x in alphas for y in gammas]


def main():
    run_suite()

if __name__ == "__main__":
    main()

