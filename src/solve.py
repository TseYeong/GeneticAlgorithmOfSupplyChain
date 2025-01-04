from solver import SupplyChainSolver
from genetic import GeneticAlgorithm

import pandas as pd
import numpy as np


def solve_boundary(instance, opt_type):
    solver = SupplyChainSolver(instance, opt_type)
    model = solver.set_equations()
    max_mod = model.generate_objective('max').solve()
    min_mod = model.generate_objective('min').solve()
    max_val = max_mod.ObjVal
    min_val = min_mod.ObjVal

    return max_val, min_val


def solve():
    pass


def save_to_file(data, path):
    pass


if __name__ == '__main__':
    instances = [
        'SS1', 'SS2', 'SS3', 'SS4', 'SS5',
        'MS1', 'MS2', 'MS3', 'MS4', 'MS5',
        'LS1', 'LS2', 'LS3', 'LS4', 'LS5',
        'ELS1', 'ELS2', 'ELS3', 'ELS4', 'ELS5'
    ]
    indices = [
        'max_cost', 'min_cost', 'max_reli', 'min_reli', 'max_flex', 'min_flex'
    ]

    boundary = pd.DataFrame(index=indices, columns=instances)

    for ins in instances:
        max_cost, min_cost = solve_boundary(ins, 'cost')
        max_reli, min_reli = solve_boundary(ins, 'reli')
        max_flex, min_flex = solve_boundary(ins, 'flex')

        boundary[ins].iloc[0] = max_cost
        boundary[ins].iloc[1] = min_cost
        boundary[ins].iloc[2] = max_reli
        boundary[ins].iloc[3] = min_reli
        boundary[ins].iloc[4] = max_flex
        boundary[ins].iloc[5] = min_flex

    path = './instances/boundary.csv'
    boundary.to_csv(path)
