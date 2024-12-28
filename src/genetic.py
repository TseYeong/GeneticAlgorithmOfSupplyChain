import random
import pandas as pd
import os


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

    def decode(self, chromosomes: list):

        Dj = [0] * self.K
        for l in range(self.L):
            chosen_index = chromosomes[-1][l] - 1
            Dj[chosen_index] += self.D[l]



