from genetic import GeneticAlgorithm

import pandas as pd
import numpy as np

def solve():
    boundary_data = pd.read_csv('./instances/boundary.csv')
    columns = boundary_data.columns

    for instance in columns:
        mac = boundary_data[instance].iloc[0]
        mic = boundary_data[instance].iloc[1]
        mar = boundary_data[instance].iloc[2]
        mir = boundary_data[instance].iloc[3]
        maf = boundary_data[instance].iloc[4]
        mif = boundary_data[instance].iloc[5]

        fit_coff = [0.5, 0.25, 0.25]

        ga = GeneticAlgorithm(instance, mac, mic, mar, mir, maf, mif)

