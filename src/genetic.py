import random
import pandas as pd
import numpy as np
import os
from list_tools import Tools


class GeneticAlgorithm:
    def __init__(self, instance, min_cost, max_cost, min_reli, max_reli, min_flex, max_flex,
                 population_size=300, generation=300, cross_p=0.8, mutation_p=0.3, fit_coff=None):
        """
        Initialization function.

        :param min_cost: Minimal cost calculated by solver.
        :type min_cost: float
        :param max_cost: Maximal cost calculated by solver.
        :type max_cost: float
        :param min_reli: Minimal reliability calculated by solver.
        :type min_reli: float
        :param max_reli: Maximal reliability calculated by solver.
        :type max_reli: float
        :param min_flex: Minimal flexibility calculated by solver.
        :type min_flex: float
        :param max_flex: Maximal flexibility calculated by solver
        :type max_flex: float
        :param instance: A string representing the name of current instance.
        :type instance: str
        :param population_size: Population size of the genetic algorithm (Default: 300).
        :type population_size: int
        :param generation: Generation size of the genetic algorithm (Default: 300).
        :type generation: int
        :param cross_p: Probability of the cross (Default: 0.8).
        :type cross_p: float
        :param mutation_p: Probability of the mutation (Default: 0.3).
        :type mutation_p: float
        """
        if fit_coff is None:
            fit_coff = [0.33, 0.33, 0.33]
        elif type(fit_coff) is not list:
            raise ValueError(f"'fit_coff' must be type of list, but got type of {type(fit_coff)} instead.")
        elif len(fit_coff) != 3:
            raise ValueError(f"'fit_coff' must be length of 3, but got length of {len(fit_coff)} instead.")

        self.instance = instance
        self.min_cost = min_cost
        self.max_cost = max_cost
        self.min_reli = min_reli
        self.max_reli = max_reli
        self.min_flex = min_flex
        self.max_flex = max_flex
        self.population_size = population_size
        self.generation = generation
        self.cross_p = cross_p
        self.mutation_p = mutation_p
        self.fit_coff = fit_coff

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
        :return: Matrix containing preference scores in stage 1 and stage 2.
        :rtype: tuple[list[list[float]], list[list[float]]]
        """
        cost = [self.Ssp, self.Spd]
        reli = [self.rsp, self.rpd]
        flex = [self.fsp, self.fpd]

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

    def decode(self, Individual):
        """
        Function of decoding of genetic algorithm.

        :param Individual: List of chromosomes of each entity in each stage.
        :type Individual: list[list[int]]
        :return: Transport matrix in each stage.
        :rtype: tuple[list, list, list]
        """
        pre_matrix_sp, pre_matrix_pd = self.preference_matrix()
        Dd = [0] * self.K
        quantity_dc = [[0 for _ in range(self.L)] for _ in range(self.K)]
        for l in range(self.L):
            chosen_index = Individual[-1][l] - 1
            Dd[chosen_index] += self.D[l]
            quantity_dc[chosen_index][l] += self.D[l]

        Cp = self.Cp.copy()
        chromosomes_pd = Individual[1].copy()
        quantity_pd = self.stage_decode(pre_matrix_pd, chromosomes_pd, Dd, Cp)

        Dp = [0] * self.J
        for j in range(self.J):
            Dp[j] = sum(quantity_pd[j])

        Cs = self.Cs.copy()
        chromosomes_sp = Individual[0].copy()
        quantity_sp = self.stage_decode(pre_matrix_sp, chromosomes_sp, Dp, Cs)

        return quantity_sp, quantity_pd, quantity_dc

    def calculate_fitness(self, quant_sp, quant_pd, quant_dc):
        """
        Function of calculating fitness value.

        :param quant_sp: Quantity matrix between suppliers to plants.
        :type quant_sp: list
        :param quant_pd: Quantity matrix between plants to distribution centers.
        :type quant_pd: list
        :param quant_dc: Quantity matrix between distribution centers to customer zones.
        :type quant_dc: list
        :return: Fitness value of given quantity matrices.
        :rtype: tuple[float, float, float, float]
        """
        cost = 0.0
        Rp = []
        Fp = []
        Fsp = [[0.0 for _ in range(self.J)] for _ in range(self.I)]

        for j in range(self.J):
            expr = 1.0
            for i in range(self.I):
                cost += self.Ssp[i][j] * quant_sp[i][j]

                if quant_sp[i][j] > 0:
                    expr *= 1 - self.rsp[i][j] * self.rs[i]

                if self.fs[i] < self.fsp[i][j]:
                    Fsp[i][j] = self.fs[i]
                else:
                    Fsp[i][j] = self.fsp[i][j]

            pla_reli = self.rp[j] * (1 - expr)
            pla_flex = sum(
                quant_sp[i][j] * Fsp[i][j] for i in range(self.I)
            ) / (sum(
                quant_sp[i][j] for i in range(self.I)
            ) + 1e-6)
            Rp.append(pla_reli)
            Fp.append(pla_flex)

            if sum(quant_pd[j]) > 0:
                cost += self.FCd[j]

        Rd = []
        Fd = []
        Fpd = [[0.0 for _ in range(self.K)] for _ in range(self.J)]

        for k in range(self.K):
            expr = 1.0
            for j in range(self.J):
                cost += self.Spd[j][k] * quant_pd[j][k]

                if quant_pd[j][k] > 0:
                    expr *= 1 - self.rpd[j][k] * Rp[j]

                if Fp[j] <= self.fpd[j][k]:
                    Fpd[j][k] = Fp[j]
                else:
                    Fpd[j][k] = self.fpd[j][k]

            dc_reli = self.rd[k] * (1 - expr)
            dc_flex = sum(
                quant_pd[j][k] * Fpd[j][k] for j in range(self.J)
            ) / (sum(
                quant_pd[j][k] for j in range(self.J)
            ) + 1e-6)

            Rd.append(dc_reli)
            Fd.append(dc_flex)

            if sum(quant_dc[k]) > 0:
                cost += self.FCd[k]

        Rc = []
        Fc = []
        Fdc = [[0.0 for _ in range(self.L)] for _ in range(self.K)]

        for l in range(self.L):
            expr = 1.0
            for k in range(self.K):
                cost += self.Sdc[k][l] * quant_dc[k][l]

                if quant_dc[k][l] > 0:
                    expr *= 1 - self.rdc[k][l] * Rd[k]

                if Fd[k] <= self.fdc[k][l]:
                    Fdc[k][l] = Fd[k]
                else:
                    Fdc[k][l] = self.fdc[k][l]

            cz_reli = 1 - expr
            cz_flex = sum(
                quant_dc[k][l] * Fdc[k][l] for k in range(self.K)
            ) / (sum(
                quant_dc[k][l] for k in range(self.K)
            ) + 1e-6)

            Rc.append(cz_reli)
            Fc.append(cz_flex)

        reliability = sum(Rc) / len(Rc)
        flexibility = sum(Fc) / len(Rc)

        cost_norm = (self.max_cost - cost) / (self.max_cost - self.min_cost)  # Minimize cost
        reli_norm = (reliability - self.min_reli) / (self.max_reli - self.min_reli)  # Maximize reliability
        flex_norm = (flexibility - self.min_flex) / (self.max_flex - self.min_flex)  # Maximize flexibility

        fitness_val = self.fit_coff[0] * cost_norm + self.fit_coff[1] * reli_norm + self.fit_coff[2] * flex_norm

        return fitness_val, cost, reliability, flexibility

    def select(self, fitness_scores):
        select_prob = []

        for fit_val in fitness_scores:
            select_prob.append(fit_val / sum(fitness_scores))

        select_index = np.random.choice(len(fitness_scores), size=self.population_size, replace=False, p=select_prob)
        return select_index

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

    def run(self):
        populations = self.initialize_population()
        best_vals = []

        for generation in range(self.generation + 1):
            fitness_scores = []
            for individual in populations:
                quant_sp, quant_pd, quant_dc = self.decode(individual)
                fit_val, _, _, _ = self.calculate_fitness(quant_sp, quant_pd, quant_dc)
                fitness_scores.append(fit_val)

            best_fit_val = max(fitness_scores)
            best_vals.append(best_fit_val)
            print(f"===>Generation {generation}, best fitness value is {best_fit_val}")

            if generation == self.generation:
                best_fit_id = fitness_scores.index(best_fit_val)
                best_individual = populations[best_fit_id]
                quant_sp, quant_pd, quant_dc = self.decode(best_individual)
                Xp = [1 if sum(quant_pd[j]) > 0 else 0 for j in range(self.J)]
                Yd = [1 if sum(quant_dc[k]) > 0 else 0 for k in range(self.K)]
                _, cost, reliability, flexibility = self.calculate_fitness(quant_sp, quant_pd, quant_dc)
                return best_vals, cost, reliability, flexibility, Xp, Yd

            # Select
            if len(populations) > self.population_size:
                select_indices = self.select(fitness_scores).tolist()
                select_indices.sort(reverse=True)
                populations_new = [populations.pop(i) for i in select_indices]
                populations = populations_new
                del populations_new

            # Crossover
            populations_id = list(range(self.population_size))
            while len(populations_id) >= 2:
                populations_id, ids = Tools.random_remove(populations_id)
                parent1 = populations[ids[0]]
                parent2 = populations[ids[1]]
                child = self.crossover(parent1, parent2)
                if child:
                    populations.append(child)

            # Mutation
            populations_id = list(range(self.population_size))
            while len(populations_id):
                populations_id, ids = Tools.random_remove(populations_id, num=1)
                parent = populations[ids[0]]
                child = self.mutate(parent)
                if child:
                    populations.append(child)
