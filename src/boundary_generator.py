from solver import SupplyChainSolver
import numpy as np
import pandas as pd


def solve_boundary(instance, opt_type, obj):
    solver = SupplyChainSolver(instance, opt_type)
    model = solver.set_equations().generate_objective(obj).solve()
    print(f"{obj.upper()} boundary of {opt_type} of instance {instance} solved.")

    return model.ObjVal


def save_to_file(data, path):
    data.to_csv(path)


def main():
    instances = [
        'SS1', 'SS2', 'SS3', 'SS4', 'SS5',
        'MS1', 'MS2', 'MS3', 'MS4', 'MS5',
        'LS1', 'LS2', 'LS3', 'LS4', 'LS5',
        'ELS1', 'ELS2', 'ELS3', 'ELS4', 'ELS5'
    ]
    indices = [
        'max_cost', 'min_cost', 'max_reli', 'min_reli', 'max_flex', 'min_flex'
    ]

    boundary_data = pd.DataFrame(index=indices, columns=instances)

    for ins in instances:
        mac = solve_boundary(ins, 'cost', 'max')
        mic = solve_boundary(ins, 'cost', 'min')
        mar = solve_boundary(ins, 'reli', 'max')
        mir = solve_boundary(ins, 'reli', 'min')
        maf = solve_boundary(ins, 'flex', 'max')
        # mif = solve_boundary(ins, 'flex', 'min')
        mif = 0.0

        bound_val = [mac, mic, mar, mir, maf, mif]

        for i in range(boundary_data.shape[0]):
            boundary_data[ins][i] = bound_val[i]

        print(f"Finish solving boundaries for instance {ins}")

    path = './instances/boundary.csv'
    save_to_file(boundary_data, path)


if __name__ == "__main__":
    main()

