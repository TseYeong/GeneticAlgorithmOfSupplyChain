import random
import pandas as pd
import numpy as np
import os
from list_tools import Tools
from solver import SupplyChainSolver


class GeneticAlgorithm:
    def __init__(self, instance: str, population_size: int = 300,
                 generation: int = 300, cross_p: float = 0.8, mutation_p: float = 0.3):

        self.instance = instance
        self.population_size = population_size
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

    def generate_boundary(self, opt_type):
        """
        Function of generating the boundary (both upper and lower) of specific optimization type based on Gurobi.

        :param opt_type: Optimization problem type.
        :type opt_type: str
        :return: A tuple containing upper and lower boundaries.
        :rtype: tuple[float, float]
        """
        solver = SupplyChainSolver(instance=self.instance, opt_type=opt_type)


    def generate_chromosome(self, stage: int):
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

    def initialize_population(self):
        """
        Initialize the population based on the population size.
        :return: List of chromosomes of each individual.
        :rtype: list[list[list[int]]]
        """
        populations = []
        for _ in range(self.population_size):
            chromosomes_list = [
                self.generate_chromosome(1),
                self.generate_chromosome(2),
                self.generate_chromosome(3)
            ]

            populations.append(chromosomes_list)

        return populations

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
        pre_matrix_pd = [[0.0 for _ in range(self.K)] for _ in range(self.J)]

        for i in range(self.I):
            for j in range(self.J):
                pre_matrix_sp[i][j] = Nrsp[i][j] + Nfsp[i][j] - Ncsp[i][j]

        for j in range(self.J):
            for k in range(self.K):
                pre_matrix_pd[j][k] = Nrpd[j][k] + Nfpd[j][k] - Ncpd[j][k]

        return pre_matrix_sp, pre_matrix_pd

    @staticmethod
    def stage_decode(pre_matrix: list, chromosomes: list, demand: list, capacity: list):
        """
        Function of decoding stage 1 and stage 2.

        :param pre_matrix: Preference matrix.
        :type pre_matrix: list[list[float]]
        :param chromosomes: List of chromosomes of each entity.
        :type chromosomes: list[int]
        :param demand: List of demand of destination entity.
        :type demand: list[int]
        :param capacity: List of capacity of start entity.
        :type capacity: list[int]
        :return: Transport matrix.
        :rtype: list[list[int]]
        """
        sta_len = len(pre_matrix)
        dest_len = len(pre_matrix[0])
        trans_quantity = [[0 for _ in range(dest_len)] for _ in range(sta_len)]

        while sum(demand):
            chrom_index, _ = Tools.find_max_value(chromosomes)

            if chrom_index >= sta_len:  # Entity is destination
                dest_index = chrom_index - sta_len

                if demand[dest_index] < 1e-6:
                    chromosomes[chrom_index] = 0
                    continue

                sta_index, _ = Tools.find_max_value(np.array(pre_matrix)[:, dest_index].tolist())

                if capacity[sta_index] >= demand[dest_index]:
                    trans_quantity[sta_index][dest_index] += demand[dest_index]
                    capacity[sta_index] -= demand[dest_index]
                    demand[dest_index] = 0

                    tmp = np.array(pre_matrix)
                    tmp[:, dest_index] = -1
                    pre_matrix = tmp.tolist()

                    chromosomes[chrom_index] = 0

                else:
                    trans_quantity[sta_index][dest_index] += capacity[sta_index]
                    demand[dest_index] -= capacity[sta_index]
                    capacity[sta_index] = 0

                    tmp = np.array(pre_matrix)
                    tmp[sta_index, :] = -1
                    pre_matrix = tmp.tolist()

            else:  # Entity is start
                if capacity[chrom_index] < 1e-6:
                    chromosomes[chrom_index] = 0
                    continue

                dest_index, _ = Tools.find_max_value(pre_matrix[chrom_index])

                if capacity[chrom_index] >= demand[dest_index]:
                    trans_quantity[chrom_index][dest_index] += demand[dest_index]
                    capacity[chrom_index] -= demand[dest_index]
                    demand[dest_index] = 0

                    tmp = np.array(pre_matrix)
                    tmp[:, dest_index] = -1
                    pre_matrix = tmp.tolist()

                else:
                    trans_quantity[chrom_index][dest_index] += capacity[chrom_index]
                    demand[dest_index] -= capacity[chrom_index]
                    capacity[chrom_index] = 0

                    tmp = np.array(pre_matrix)
                    tmp[chrom_index, :] = -1
                    pre_matrix = tmp.tolist()

                    chromosomes[chrom_index] = 0

        return trans_quantity

    def decode(self, population):
        """
        Function of decoding of genetic algorithm.

        :param population: List of chromosomes of each entity in each stage.
        :type population: list[list[int]]
        :return: Transport matrix in stage 1 and stage 2.
        :rtype: tuple[list, list]
        """
        pre_matrix_sp, pre_matrix_pd = self.preference_matrix()
        Dd = [0] * self.K
        for l in range(self.L):
            chosen_index = population[-1][l] - 1
            Dd[chosen_index] += self.D[l]

        Cp = self.Cp.copy()
        chromosomes_pd = population[1].copy()
        quantity_pd = self.stage_decode(pre_matrix_pd, chromosomes_pd, Dd, Cp)

        Dp = [0] * self.J
        for j in range(self.J):
            Dp[j] = sum(quantity_pd[j])

        Cs = self.Cs.copy()
        chromosomes_sp = population[0].copy()
        quantity_sp = self.stage_decode(pre_matrix_sp, chromosomes_sp, Dp, Cs)

        return quantity_sp, quantity_pd

    def select(self):
        pass

    def crossover(self, parent1, parent2):
        """
        Crossover operator of the generation.

        :param parent1: List of chromosomes of the 1st parent.
        :type parent1: list[list[int]]
        :param parent2: List of chromosomes of the 2nd parent.
        :type parent2: list[list[int]]
        :return: List of chromosomes of the children after crossover.
        :rtype: list[list[int]]
        """
        if np.random.rand() < self.cross_p:
            segment = np.random.randint(2, size=3)
            child = []

            for index, binary in enumerate(segment):
                if binary:
                    child.append(parent2[index].copy())
                else:
                    child.append(parent1[index].copy())

            return child

        return

    def mutate(self, individual):
        """
        Mutation operator of the generation.

        :param individual: List of chromosomes of the individual.
        :type individual: list[list[int]]
        :return: List of chromosomes of the children after mutation.
        :rtype: list[list[int]]
        """
        if np.random.rand() < self.mutation_p:
            segment = np.random.randint(2, size=3)
            child = []
            for index, binary in enumerate(segment):
                if binary:
                    chromosomes = Tools.element_swap(individual[index])
                    child.append(chromosomes)
                else:
                    child.append(individual[index].copy())

            return child

        return
