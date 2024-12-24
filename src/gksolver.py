import pandas as pd
from gekko import GEKKO


class CostOptimization:

    def __init__(self, df_faci_params: pd.DataFrame, df_stage1: pd.DataFrame,
                 df_stage2: pd.DataFrame, df_stage3: pd.DataFrame, obj: str = 'min') -> None:
        """
        Initialization function.

        :param df_faci_params: A DataFrame containing the parameters of each facility.
        :type df_faci_params: pd.DataFrame
        :param df_stage1: A DataFrame containing the shipping parameters between suppliers and plants.
        :type df_stage1: pd.DataFrame
        :param df_stage2: A DataFrame containing the shipping parameters between plants and distribution centers.
        :type df_stage2: pd.DataFrame
        :param df_stage3: A DataFrame containing the shipping parameters between distribution centers and customer zones.
        :type df_stage3: pd.DataFrame
        :param obj: Objective (min, max).
        :type obj: str
        """
        self.df_faci_params = df_faci_params
        self.df_stage1 = df_stage1
        self.df_stage2 = df_stage2
        self.df_stage3 = df_stage3
        self.obj = obj

        self.m = GEKKO(remote=True)
        self.m.options.MAX_MEMORY = 5

        # Number of facilities
        self.I = df_faci_params['sup_cap'].count()  # Number of suppliers
        self.J = df_faci_params['pla_cap'].notna().sum()  # Number of plants
        self.K = df_faci_params['dc_cap'].notna().sum()  # Number of distribution centers
        self.L = df_stage3.shape[1]  # Number of customer zones

        # Constant parameters
        self.Cs = df_faci_params['Sup_Cap'].to_list()  # Capacity of suppliers
        self.Cp = df_faci_params['Pla_Cap'].to_list()  # Capacity of plants
        self.Cd = df_faci_params['DC_Cap'].to_list()  # Capacity of distribution centers
        self.D = list(map(int, df_stage3.iloc[-2].to_list()))  # Demand at customer zones
        self.SL = [float(x.strip('%')) / 100 for x in df_stage3.iloc[-1].to_list()]  # Service level at customer zones

        # Fixed cost
        self.FCp = df_faci_params['Pla_Fix_Cost'].dropna().to_list()
        self.FCd = df_faci_params['DC_Fix_Cost'].dropna().to_list()

        # Shipping cost
        self.Ssp = [[float(x.split(',')[0]) for x in df_stage1.iloc[i]] for i in range(self.I)]
        self.Spd = [[float(x.split(',')[0]) for x in df_stage2.iloc[j]] for j in range(self.J)]
        self.Sdc = [[float(x.split(',')[0]) for x in df_stage3.iloc[k]] for k in range(self.K)]

        # Initialize variables
        self.Qsp = [[self.m.Var(lb=0, integer=True) for _ in range(self.J)] for _ in range(self.I)]
        self.Qpd = [[self.m.Var(lb=0, integer=True) for _ in range(self.K)] for _ in range(self.J)]
        self.Qdc = [[self.m.Var(lb=0, integer=True) for _ in range(self.L)] for _ in range(self.K)]
        self.qdc = [[self.m.Var(lb=0, ub=1, integer=True) for _ in range(self.L)] for _ in range(self.K)]
        self.X = [self.m.Var(lb=0, ub=1, integer=True) for _ in range(self.J)]
        self.Y = [self.m.Var(lb=0, ub=1, integer=True) for _ in range(self.K)]

    def set_equations(self):
        """
        Function of setting up the constraints for the cost optimization model.

        Constraints include:
        - Supplier capacity constraint.
        - Plant capacity constraint.
        - DC capacity constraint.
        - Demand meeting constraint.
        - Material flow constraint.
        - Single sourcing constraint.
        """
        # Constraint 1
        for i in range(self.I):
            self.m.Equation(sum(self.Qsp[i][j] for j in range(self.J)) <= self.Cs[i])

        # Constraint 2
        for j in range(self.J):
            self.m.Equation(sum(self.Qpd[j][k] for k in range(self.K)) <= self.Cp[j] * self.X[j])

        # Constraint 3
        for k in range(self.K):
            self.m.Equation(sum(self.Qdc[k][l] for l in range(self.L)) <= self.Cd[k] * self.Y[k])

        # Constraint 4
        for l in range(self.L):
            self.m.Equation(sum(self.Qdc[k][l] for k in range(self.K)) >= self.D[l] * self.SL[l])

        # Constraint 5
        for j in range(self.J):
            self.m.Equation(sum(self.Qsp[i][j] for i in range(self.I)) == sum(self.Qpd[j][k] for k in range(self.K)))
        for k in range(self.K):
            self.m.Equation(sum(self.Qpd[j][k] for j in range(self.J)) == sum(self.Qdc[k][l] for l in range(self.L)))

        # Constraint 6
        for l in range(self.L):
            self.m.Equation(sum(self.qdc[k][l] for k in range(self.K)) == 1)
        # Big M method (to be implemented)

    def generate_objective(self):
        """
        Function of generating optimization objective.

        Costs include:
        - Fixed costs for activating plants and distribution centers
        - Transportation costs from suppliers to plants
        - Transportation costs from plants to distribution centers
        - Transportation costs from distribution centers to customer zones
        """
        # Intermediate variable
        fixed_cost_p = self.m.Intermediate(sum(self.FCp[j] * self.X[j] for j in range(self.J)))
        fixed_cost_dc = self.m.Intermediate(sum(self.FCd[k] * self.Y[k] for k in range(self.K)))

        cost_sp_list = []
        for i in range(self.I):
            cost_sp_i = self.m.Intermediate(sum(self.Ssp[i][j] * self.Qsp[i][j] for j in range(self.J)))
            cost_sp_list.append(cost_sp_i)
        cost_sp = self.m.Intermediate(sum(cost_sp_list))

        cost_pd_list = []
        for j in range(self.J):
            cost_pd_j = self.m.Intermediate(sum(self.Spd[j][k] * self.Qpd[j][k] for k in range(self.K)))
            cost_pd_list.append(cost_pd_j)
        cost_pd = self.m.Intermediate(sum(cost_pd_list))

        cost_dc_list = []
        for k in range(self.K):
            cost_dc_k = self.m.Intermediate(sum(self.Sdc[k][l] * self.Qdc[k][l] for l in range(self.L)))
            cost_dc_list.append(cost_dc_k)

        cost_dc = self.m.Intermediate(sum(cost_dc_list))

        # Objective function
        if self.obj.lower() == 'min':
            self.m.Minimize(fixed_cost_p + fixed_cost_dc + cost_sp + cost_pd + cost_dc)
        elif self.obj.lower() == 'max':
            self.m.Maximize(fixed_cost_p + fixed_cost_dc + cost_sp + cost_pd + cost_dc)
        else:
            raise ValueError("Invalid value for 'obj'. "
                             "Expected 'min' or 'max', but got '{}'.".format(self.obj))

    def solve(self):
        """
        Solves the optimization problem using the GEKKO solver.
        """
        self.m.solve()
