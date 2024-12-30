import random
import pandas as pd
import os
from list_tools import Tools
import numpy as np


class GeneticAlgorithm:
    def __init__(self, instance: str, population: int = 300,
                 generation: int = 300, cross_p: float = 0.8, mutation_p: float = 0.3):

        self.instance = instance
        self.population = population
        self.generation = generation
        self.cross_p = cross_p
        self.mutation_p = mutation_p

        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df_faci_params = pd.read_csv(f"{root_path}/instances/{self.instance}/facilities_params_{self.instance}.csv")
        df_stage1 = pd.read_csv(f"{root_path}/instances/{self.instance}/sup2pla_cost_{self.instance}.csv")
        df_stage2 = pd.read_csv(f"{root_path}/instances/{self.instance}/pla2dc_cost_{self.instance}.csv")
        df_stage3 = pd.read_csv(f"{root_path}/instances/{self.instance}/dc2cz_cost_{self.instance}.csv")

        # Number of facilities
        self.I = int(df_faci_params['sup_cap'].count())  # Number of suppliers
        self.J = int(df_faci_params['pla_cap'].notna().sum())  # Number of plants
        self.K = int(df_faci_params['dc_cap'].notna().sum())  # Number of distribution centers
        self.L = int(df_stage3.shape[1])  # Number of customer zones

        # Constant parameters
        self.Cs = df_faci_params['sup_cap'].to_list()  # Capacity of suppliers
        self.Cp = df_faci_params['pla_cap'].dropna().to_list()  # Capacity of plants
        self.Cd = df_faci_params['dc_cap'].dropna().to_list()  # Capacity of distribution centers
        self.D = list(map(int, df_stage3.iloc[-2].to_list()))  # Demand at customer zones
        self.SL = [float(x.strip('%')) / 100 for x in df_stage3.iloc[-1].to_list()]  # Service level at customer zones

        self.rs = df_faci_params['sup_reli'].to_list()  # Reliability index of suppliers.
        self.rp = df_faci_params['pla_reli'].dropna().to_list()  # Reliability index of plants.
        self.rd = df_faci_params['dc_reli'].dropna().to_list()  # Reliability index of distribution centers.
        self.fs = df_faci_params['sup_flex'].to_list()  # Flexibility index of suppliers.

        # Fixed cost
        self.FCp = df_faci_params['pla_cost'].dropna().to_list()
        self.FCd = df_faci_params['dc_cost'].dropna().to_list()

        # Shipping cost
        self.Ssp = [[float(x.split(',')[0]) for x in df_stage1.iloc[i]] for i in range(self.I)]
        self.Spd = [[float(x.split(',')[0]) for x in df_stage2.iloc[j]] for j in range(self.J)]
        self.Sdc = [[float(x.split(',')[0]) for x in df_stage3.iloc[k]] for k in range(self.K)]

        # Reliability of arc
        self.rsp = [[float(x.split(',')[1]) for x in df_stage1.iloc[i]] for i in range(self.I)]
        self.rpd = [[float(x.split(',')[1]) for x in df_stage2.iloc[j]] for j in range(self.J)]
        self.rdc = [[float(x.split(',')[1]) for x in df_stage3.iloc[k]] for k in range(self.K)]

        # Volume flexibility of link
        self.fsp = [[float(x.split(',')[2]) for x in df_stage1.iloc[i]] for i in range(self.I)]
        self.fpd = [[float(x.split(',')[2]) for x in df_stage2.iloc[j]] for j in range(self.J)]
        self.fdc = [[float(x.split(',')[2]) for x in df_stage3.iloc[k]] for k in range(self.K)]

    def generate_chromosome(self, stage: int) -> list:
        """
        Generate chromosome based on the stages.

        :param stage: Stage where chromosome need to be generated.
        :type stage: int
        :return: List containing chromosomes.
        :rtype: list
        """
        if stage == 3:
            chromosomes = [-1] * self.L
            capacity = self.Cd.copy()

            for l in range(self.L):
                valid_index = [k for k in range(self.K) if capacity[k] >= self.D[l]]
                chosen_index = random.choice(valid_index)

                chromosomes[l] = chosen_index + 1
                capacity[chosen_index] -= self.D[l]

            return chromosomes

        elif stage == 2:
            num = self.J + self.K

        else:
            num = self.I + self.J

        return random.sample(range(1, num + 1), num)

    def preference_matrix(self):
        """
        Generate preference matrix in stage 1 and 2.
        :return:
        :rtype: tuple[list[list[float]], list[list[float]]]
        """
        cost = [self.Ssp, self.Spd, self.Sdc]
        reli = [self.rsp, self.rpd, self.rdc]
        flex = [self.fsp, self.fpd, self.fdc]

        normalized_cost = Tools.normalization(cost)
        normalized_reli = Tools.normalization(reli)
        normalized_flex = Tools.normalization(flex)

        Ncsp = normalized_cost[0]
        Ncpd = normalized_cost[1]
        Nrsp = normalized_reli[0]
        Nrpd = normalized_reli[1]
        Nfsp = normalized_flex[0]
        Nfpd = normalized_flex[1]

        pre_matrix_sp = [[0.0 for _ in range(self.J)] for _ in range(self.I)]
        print(pre_matrix_sp[1][1])
        pre_matrix_pd = [[0.0 for _ in range(self.K)] for _ in range(self.J)]

        for i in range(self.I):
            for j in range(self.J):
                pre_matrix_sp[i][j] = Nrsp[i][j] + Nfsp[i][j] - Ncsp[i][j]

        for j in range(self.J):
            for k in range(self.K):
                pre_matrix_pd[j][k] = Nrpd[j][k] + Nfpd[j][k] - Ncpd[j][k]

        return pre_matrix_sp, pre_matrix_pd

    def decode(self, chromosomes: list):

        Dj = [0] * self.K
        for l in range(self.L):
            chosen_index = chromosomes[-1][l] - 1
            Dj[chosen_index] += self.D[l]

        index, chromosome = Tools.find_max_value(chromosomes[1])
        pre_matrix_sp, pre_matrix_pd = self.preference_matrix()
        if index >= self.J:  # Entity is DC
            dc_index = index - self.J
            row, pre_value = Tools.find_max_value(np.array(pre_matrix_pd)[:, dc_index].tolist())
