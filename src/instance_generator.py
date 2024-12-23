import numpy as np
import pandas as pd
import os


def generate_capacity(num_fac: int) -> list:
    """
    Function of generating the capacity of facilities.

    :param num_fac: Number of facilities.
    :type num_fac: int
    :return: An list containing the generated capacity.
    :rtype: list
    """
    return list(np.random.randint(2000, 40001, num_fac))


def generate_reliability(num_fac: int) -> list:
    """
    Function of generating the reliability of facilities.

    :param num_fac: Number of facilities.
    :type num_fac: int
    :return: An list containing the generated reliability.
    :rtype: list
    """
    return list(np.round(np.random.rand(num_fac) * 0.15 + 0.8, 2))


def generate_flexibility(num_fac: int) -> list:
    """
    Function of generating the flexibility of facilities.

    :param num_fac: Number of facilities.
    :type num_fac: int
    :return: An list containing the generated flexibility.
    :rtype: list
    """
    return list(np.round(np.random.rand(num_fac) * 0.25 + 0.6, 2))


def generate_cost(num_fac: int) -> list:
    """
    Function of generating the cost of facilities
    :param num_fac: umber of facilities.
    :type num_fac: int
    :return: An list containing the generated cost.
    :rtype: list
    """
    return list(np.random.randint(5000, 15001, num_fac))


def generate_shipping_cost() -> float:
    """
    Function of generating the shipping cost per unit item between two entities.

    :return: A float number representing the shipping cost.
    :rtype: float
    """
    return round(np.random.rand() * 3 + 1, 1)


def generate_shipping_reliability() -> float:
    """
    Function of generating the reliability of arc.

    :return: A float number representing the reliability of arc.
    :rtype: float
    """
    return round(np.random.rand() * 0.2 + 0.7, 2)


def generate_shipping_flexibility() -> float:
    """
    Function of generating the volume flexibility.

    :return: A float number representing the volume flexibility.
    :rtype: float
    """
    return round(np.random.rand() * 0.35 + 0.45, 2)


def generate_demand() -> int:
    """
    FUnction of generating the demand of any customer zone.
    :return: An integer containing the demand.
    :rtype: int
    """
    return np.random.randint(200, 501)


def generate_facility_params(num_sup: int, num_pla: int, num_dc: int) -> dict:
    """
    Function of generating the capacity, reliability, flexibility,
    and cost parameters for suppliers, plants, and distribution centers.

    :param num_sup: Number of suppliers.
    :type num_sup: int
    :param num_pla: Number of plants.
    :type num_pla: int
    :param num_dc: Number of distribution centers.
    :type num_dc: int
    :return: A dictionary containing the generated parameters.
    :rtype: dict
    """

    # Suppliers params
    sup_cap = generate_capacity(num_sup)
    sup_reli = generate_reliability(num_sup)
    sup_flex = generate_flexibility(num_sup)

    # Plats params
    pla_cap = generate_capacity(num_pla)
    pla_reli = generate_reliability(num_pla)
    pla_cost = generate_cost(num_pla)

    # Distribution Center params
    dc_cap = generate_capacity(num_dc)
    dc_reli = generate_reliability(num_dc)
    dc_cost = generate_cost(num_dc)

    # Extend none values with 'N/A' to match the number of suppliers
    pla_cap.extend(['N/A'] * (num_sup - num_pla))
    pla_reli.extend(['N/A'] * (num_sup - num_pla))
    pla_cost.extend(['N/A'] * (num_sup - num_pla))

    dc_cap.extend(['N/A'] * (num_sup - num_dc))
    dc_reli.extend(['N/A'] * (num_sup - num_dc))
    dc_cost.extend(['N/A'] * (num_sup - num_dc))

    # Return the params dictionary
    return {
        'sup_cap': sup_cap,
        'sup_reli': sup_reli,
        'sup_flex': sup_flex,
        'pla_cap': pla_cap,
        'pla_reli': pla_reli,
        'pla_cost': pla_cost,
        'dc_cap': dc_cap,
        'dc_reli': dc_reli,
        'dc_cost': dc_cost
    }


def generate_shipping_matrix(num_src: int, num_dst: int) -> np.ndarray:
    """
    Function of generating the shipping cost matrix between two sets of entities.

    :param num_src: Number of source entities (Suppliers, Plants).
    :type num_src: int
    :param num_dst: Number of destination entities (Plants, Distribution Centers).
    :type num_dst: int
    :return: A matrix containing the shipping cost, reliability, and flexibility.
    :rtype: np.ndarray
    """
    matrix = np.empty((num_src, num_dst), dtype=object)

    for src_id in range(num_src):
        for dst_id in range(num_dst):
            ship_cost = generate_shipping_cost()
            ship_reli = generate_shipping_reliability()
            ship_flex = generate_shipping_flexibility()
            ship_str = ",".join([str(ship_cost), str(ship_reli), str(ship_flex)])

            matrix[src_id][dst_id] = ship_str

    return matrix


def generate_dc_to_cz_matrix(num_dc: int, num_cz: int, fill_rate: str = None) -> np.ndarray:
    """
    Function of generating the shipping cost matrix between distribution centers and customer zones.

    :param num_dc: Number of distribution centers.
    :type num_dc: int
    :param num_cz: Number of customer zones.
    :type num_cz: int
    :param fill_rate: Fill rate (Service level) target at customer zone
    :type fill_rate: str
    :return: A matrix containing the shipping cost, reliability, and flexibility, along with demand and fill rate.
    :rtype: np.ndarray
    """
    dc2cz_matrix = np.empty((num_dc + 2, num_cz), dtype=object)

    for dc_id in range(num_dc):
        for cz_id in range(num_cz):
            ship_cost = generate_shipping_cost()
            ship_reli = generate_shipping_reliability()
            ship_flex = generate_shipping_flexibility()
            ship_str = ",".join([str(ship_cost), str(ship_reli), str(ship_flex)])

            dc2cz_matrix[dc_id][cz_id] = ship_str

    # Demand and fill rate
    for cz_id in range(num_cz):
        demand = generate_demand()
        dc2cz_matrix[num_dc][cz_id] = demand
        if fill_rate is None:
            dc2cz_matrix[num_dc + 1][cz_id] = '100%'
        else:
            dc2cz_matrix[num_dc + 1][cz_id] = fill_rate

    return dc2cz_matrix


def save_to_csv(data: dict | np.ndarray, file_name: str) -> None:
    """
    Function of saving a dictionary or matrix to a CSV file.

    :param data: Data to save.
    :type data: dict | np.ndarray
    :param file_name: The path to the csv file.
    :type file_name: str
    """
    df = pd.DataFrame(data)
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    df.to_csv(file_name, index=False, encoding='utf-8')


def generate_instance(prob_code: str, num_sup: int, num_pla: int, num_dc: int, num_cz: int) -> None:
    """
    Function to generate all parameters for a given instance.

    :param prob_code: Unique identifier for the problem instance (e.g., 'LS2', 'MS3').
    :type prob_code: str
    :param num_sup: Number of suppliers.
    :type num_sup: int
    :param num_pla: Number of plants.
    :type num_pla: int
    :param num_dc: Number of distribution centers.
    :type num_dc: int
    :param num_cz: Number of customer zones.
    :type num_cz: int
    """
    print(f"===> Generating instance {prob_code}")

    # Generate facility parameters
    faci_params = generate_facility_params(num_sup, num_pla, num_dc)
    write_path = f"./instances/{prob_code}/facilities_params_{prob_code}.csv"
    save_to_csv(faci_params, write_path)

    print(f"\tFinish generating parameters of facilities")

    # Generate shipping matrices for Stage 1 and Stage 2
    sup2pla_matrix = generate_shipping_matrix(num_sup, num_pla)
    write_path = f"./instances/{prob_code}/sup2pla_cost_{prob_code}.csv"
    save_to_csv(sup2pla_matrix, write_path)

    pla2dc_matrix = generate_shipping_matrix(num_pla, num_dc)
    write_path = f"./instances/{prob_code}/pla2dc_cost_{prob_code}.csv"
    save_to_csv(pla2dc_matrix, write_path)

    print("\tFinish generating parameters in stages 1 and 2")

    # Generate shipping matrix between DC and Customer Zones
    dc2cz_matrix = generate_dc_to_cz_matrix(num_dc, num_cz)
    write_path = f'./instances/{prob_code}/dc2cz_cost_{prob_code}.csv'
    save_to_csv(dc2cz_matrix, write_path)

    print("\tFinish generating parameters between DC and CZ")
    print(f"Finish generating instance {prob_code}")


def main():
    prob_codes = ['SS1', 'SS2', 'SS3', 'SS4', 'SS5',
                  'MS1', 'MS2', 'MS3', 'MS4', 'MS5',
                  'LS1', 'LS2', 'LS3', 'LS4', 'LS5',
                  'ELS1', 'ELS2', 'ELS3', 'ELS4', 'ELS5']

    num_sups = [5, 10, 10, 10, 15,
                20, 25, 30, 35, 35,
                80, 100, 120, 150, 200,
                200, 200, 200, 200, 200]

    num_plas = [2, 2, 2, 2, 3,
                3, 3, 4, 4, 4,
                4, 4, 4, 5, 5,
                10, 20, 40, 40, 50]

    num_dcs = [2, 2, 3, 4, 5,
               5, 5, 6, 6, 6,
               6, 6, 6, 7, 7,
               10, 20, 40, 40, 50]

    num_czs = [5, 5, 7, 10, 10,
               15, 20, 25, 30, 45,
               60, 80, 100, 120, 150,
               150, 150, 150, 200, 220]

    for i in range(len(prob_codes)):
        generate_instance(prob_codes[i], num_sups[i], num_plas[i], num_dcs[i], num_czs[i])


if __name__ == "__main__":
    main()

