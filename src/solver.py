import pandas as pd
from gekko import GEKKO
import gurobipy as grb


class CostOptimization:
    """
    A class for solving the supply chain cost optimization problem.

    Attributes:
        I (int): Number of suppliers.
        J (int): Number of plants.
        K (int): Number of distribution centers.
        L (int): Number of customer zones.
        Cs (list): List of supply capacities for each supplier.
        Cp (list): List of plant capacities for each plant.
        Cd (list): List of distribution center capacities for each DC.
        D (list): List of demands for each customer zone.
        SL (list): List of fill rate (service levels) for each customer.
        Qsp (list of lists): Decision variables for quantities shipped from suppliers to plants.
        Qpd (list of lists): Decision variables for quantities shipped from plants to distribution centers.
        Qdc (list of lists): Decision variables for quantities shipped from DCs to customer zones.
        qdc (list of lists): Binary decision variables indicating whether a DC is active for a CZ.
        X (list): Binary decision variables for activating plants.
        Y (list): Binary decision variables for activating distribution centers
    """
    def __init__(self, instance: str, obj: str = 'min', solver: str = 'GEKKO'):
        """
        Initialization function.

        :param instance: A string representing the name of current instance.
        :type instance: str
        :param obj: Objective (min, max).
        :type obj: str
        :param solver: The selected solver (Default: GEKKO, Option: GEKKO, Gurobi)
        :type solver: str
        """
        self.instance = instance
        self.obj = obj
        self.solver = solver

        df_faci_params = pd.read_csv(f"instances/{self.instance}/facilities_params_{self.instance}.csv")
        df_stage1 = pd.read_csv(f"instances/{self.instance}/sup2pla_cost_{self.instance}.csv")
        df_stage2 = pd.read_csv(f"instances/{self.instance}/pla2dc_cost_{self.instance}.csv")
        df_stage3 = pd.read_csv(f"instances/{self.instance}/dc2cz_cost_{self.instance}.csv")

        if solver == 'GEKKO':
            self.m = GEKKO(remote=True)
            self.m.options.MAX_MEMORY = 5
        elif solver == 'Gurobi':
            self.model = grb.Model(name="CostOptimization")
        else:
            raise ValueError("Invalid value for 'solver'. "
                             "Expected 'GEKKO' or 'Gurobi', but got '{}'.".format(self.obj))

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

        # Fixed cost
        self.FCp = df_faci_params['Pla_Fix_Cost'].dropna().to_list()
        self.FCd = df_faci_params['DC_Fix_Cost'].dropna().to_list()

        # Shipping cost
        self.Ssp = [[float(x.split(',')[0]) for x in df_stage1.iloc[i]] for i in range(self.I)]
        self.Spd = [[float(x.split(',')[0]) for x in df_stage2.iloc[j]] for j in range(self.J)]
        self.Sdc = [[float(x.split(',')[0]) for x in df_stage3.iloc[k]] for k in range(self.K)]

        # Initialize variables
        if solver == 'GEKKO':
            self.Qsp = [[self.m.Var(lb=0, integer=True) for _ in range(self.J)] for _ in range(self.I)]
            self.Qpd = [[self.m.Var(lb=0, integer=True) for _ in range(self.K)] for _ in range(self.J)]
            self.Qdc = [[self.m.Var(lb=0, integer=True) for _ in range(self.L)] for _ in range(self.K)]
            self.qdc = [[self.m.Var(lb=0, ub=1, integer=True) for _ in range(self.L)] for _ in range(self.K)]
            self.X = [self.m.Var(lb=0, ub=1, integer=True) for _ in range(self.J)]
            self.Y = [self.m.Var(lb=0, ub=1, integer=True) for _ in range(self.K)]
        else:
            self.Qsp = self.model.addVars(self.I, self.J, lb=0, vtype=grb.GRB.INTEGER, name='Qsp')
            self.Qpd = self.model.addVars(self.J, self.K, lb=0, vtype=grb.GRB.INTEGER, name='Qpd')
            self.Qdc = self.model.addVars(self.K, self.L, lb=0, vtype=grb.GRB.INTEGER, name='Qdc')
            self.qdc = self.model.addVars(self.K, self.L, vtype=grb.GRB.BINARY, name='qdc')
            self.X = self.model.addVars(self.J, vtype=grb.GRB.BINARY, name='X')
            self.Y = self.model.addVars(self.K, vtype=grb.GRB.BINARY, name='Y')

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
        if self.solver == 'GEKKO':
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
            # Logic constraint
            upper = 500
            lower = 100
            for k in range(self.K):
                for l in range(self.L):
                    self.m.Equation(self.Qdc[k][l] <= upper * self.qdc[k][l])
                    self.m.Equation(self.Qdc[k][l] >= lower * self.qdc[k][l])

        else:
            for i in range(self.I):
                self.model.addConstr(
                    grb.quicksum(
                        self.Qsp[i, j] for j in range(self.J)
                    ) <= self.Cs[i], name=f"Constraint1_{i}"
                )

            for j in range(self.J):
                self.model.addConstr(
                    grb.quicksum(
                        self.Qpd[j, k] for k in range(self.K)
                    ) <= self.Cp[j] * self.X[j], name=f"Constraint2_{j}"
                )

            for k in range(self.K):
                self.model.addConstr(
                    grb.quicksum(
                        self.Qdc[k, l] for l in range(self.L)
                    ) <= self.Cd[k] * self.Y[k], name=f"Constraint3_{k}"
                )

            for l in range(self.L):
                self.model.addConstr(
                    grb.quicksum(
                        self.Qdc[k, l] for k in range(self.K)
                    ) == self.D[l] * self.SL[l], name=f"Constraint4_{l}"
                )

            for j in range(self.J):
                self.model.addConstr(
                    grb.quicksum(
                        self.Qsp[i, j] for i in range(self.I)
                    ) == grb.quicksum(
                        self.Qpd[j, k] for k in range(self.K)
                    ), name=f"Constraint5_{j}"
                )

            for k in range(self.K):
                self.model.addConstr(
                    grb.quicksum(
                        self.Qpd[j, k] for j in range(self.J)
                    ) == grb.quicksum(
                        self.Qdc[k, l] for l in range(self.L)
                    ), name=f"Constraint6_{k}"
                )

            for k in range(self.K):
                for l in range(self.L):
                    self.model.addConstr(
                        self.Qdc[k, l] <= 500 * self.qdc[k, l], name=f"BigM_{k}_{l}"
                    )
                    self.model.addConstr(
                        self.Qdc[k, l] >= 100 * self.qdc[k, l], name=f"NonNegative_{k}_{l}"
                    )

    def generate_objective(self):
        """
        Function of generating optimization objective.

        Costs include:
        - Fixed costs for activating plants and distribution centers
        - Transportation costs from suppliers to plants
        - Transportation costs from plants to distribution centers
        - Transportation costs from distribution centers to customer zones
        """
        if self.solver == 'GEKKO':
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
        else:
            if self.obj.lower() == 'min':
                self.model.setObjective(
                    grb.quicksum(
                        self.FCp[j] * self.X[j] for j in range(self.J)
                    ) + grb.quicksum(
                        self.FCd[k] * self.Y[k] for k in range(self.K)
                    ) + grb.quicksum(
                        self.Ssp[i][j] * self.Qsp[i, j] for i in range(self.I) for j in range(self.J)
                    ) + grb.quicksum(
                        self.Spd[j][k] * self.Qpd[j, k] for j in range(self.J) for k in range(self.K)
                    ) + grb.quicksum(
                        self.Sdc[k][l] * self.Qdc[k, l] for k in range(self.K) for l in range(self.L)
                    ), grb.GRB.MINIMIZE
                )
            elif self.obj.lower() == 'max':
                self.model.setObjective(
                    grb.quicksum(
                        self.FCp[j] * self.X[j] for j in range(self.J)
                    ) + grb.quicksum(
                        self.FCd[k] * self.Y[k] for k in range(self.K)
                    ) + grb.quicksum(
                        self.Ssp[i][j] * self.Qsp[i, j] for i in range(self.I) for j in range(self.J)
                    ) + grb.quicksum(
                        self.Spd[j][k] * self.Qpd[j, k] for j in range(self.J) for k in range(self.K)
                    ) + grb.quicksum(
                        self.Sdc[k][l] * self.Qdc[k, l] for k in range(self.K) for l in range(self.L)
                    ), grb.GRB.MAXIMIZE
                )
            else:
                raise ValueError("Invalid value for 'obj'. "
                                 "Expected 'min' or 'max', but got '{}'.".format(self.obj))

    def solve(self):
        """
        Solves the optimization problem using the GEKKO solver.
        """
        if self.solver == 'GEKKO':
            self.m.solve()
        else:
            self.model.optimize()


class ReliabilityOptimization:

    def __init__(self, df_faci_params: pd.DataFrame, df_stage1: pd.DataFrame,
                 df_stage2: pd.DataFrame, df_stage3: pd.DataFrame, obj: str = 'min'):
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
        self.Cp = df_faci_params['Pla_Cap'].dropna().to_list()  # Capacity of plants
        self.Cd = df_faci_params['DC_Cap'].dropna().to_list()  # Capacity of distribution centers
        self.D = list(map(int, df_stage3.iloc[-2].to_list()))  # Demand at customer zones
        self.SL = [float(x.strip('%')) / 100 for x in df_stage3.iloc[-1].to_list()]  # Service level at customer zones

        self.rs = df_faci_params['sup_reli'].to_list()  # Reliability index of suppliers.
        self.rp = df_faci_params['pla_reli'].dropna().to_list()  # Reliability index of plants.
        self.rd = df_faci_params['dc_reli'].dropna().to_list()  # Reliability index of distribution centers.

        # Reliability of arc
        self.rsp = [[float(x.split(',')[1]) for x in df_stage1.iloc[i]] for i in range(self.I)]
        self.rpd = [[float(x.split(',')[1]) for x in df_stage2.iloc[j]] for j in range(self.J)]
        self.rdc = [[float(x.split(',')[1]) for x in df_stage3.iloc[k]] for k in range(self.K)]

        # Initialize variables
        self.Qsp = [[self.m.Var(lb=0, integer=True) for _ in range(self.J)] for _ in range(self.I)]
        self.Qpd = [[self.m.Var(lb=0, integer=True) for _ in range(self.K)] for _ in range(self.J)]
        self.Qdc = [[self.m.Var(lb=0, integer=True) for _ in range(self.L)] for _ in range(self.K)]
        self.qsp = [[self.m.Var(lb=0, ub=1, integer=True) for _ in range(self.J)] for _ in range(self.I)]
        self.qpd = [[self.m.Var(lb=0, ub=1, integer=True) for _ in range(self.K)] for _ in range(self.J)]
        self.qdc = [[self.m.Var(lb=0, ub=1, integer=True) for _ in range(self.L)] for _ in range(self.K)]

        self.X = [self.m.Var(lb=0, ub=1, integer=True) for _ in range(self.J)]
        self.Y = [self.m.Var(lb=0, ub=1, integer=True) for _ in range(self.K)]

        self.Rp = [self.m.Var(lb=0, ub=1, integer=False) for _ in range(self.J)]
        self.Rd = [self.m.Var(lb=0, ub=1, integer=False) for _ in range(self.K)]
        self.Rc = [self.m.Var(lb=0, ub=1, integer=False) for _ in range(self.L)]

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
        - Reliability computation
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
        # Logic constraint
        lower = 100
        sp_pd_upper = sum(self.D)
        for i in range(self.I):
            for j in range(self.J):
                self.m.Equation(self.Qsp[i][j] <= sp_pd_upper * self.qsp[i][j])
                self.m.Equation(self.Qsp[i][j] >= lower * self.qsp[i][j])

        for j in range(self.J):
            for k in range(self.K):
                self.m.Equation(self.Qpd[j][k] <= sp_pd_upper * self.qpd[j][k])
                self.m.Equation(self.Qpd[j][k] >= lower * self.qpd[j][k])

        dc_upper = 500
        for k in range(self.K):
            for l in range(self.L):
                self.m.Equation(self.Qdc[k][l] <= dc_upper * self.qdc[k][l])
                self.m.Equation(self.Qdc[k][l] >= lower * self.qdc[k][l])

        # Constraint 7
        for j in range(self.J):
            terms = [self.m.log(1 - self.qsp[i][j] * self.rsp[i][j] * self.rs[i] + 1e-6) for i in range(self.I)]
            self.m.Equation(self.Rp[j] == self.rp[j] * (1 - self.m.exp(self.m.sum(terms))))

        for k in range(self.K):
            terms = [self.m.log(1 - self.qpd[j][k] * self.rpd[j][k] * self.Rp[j] + 1e-6) for j in range(self.J)]
            self.m.Equation(self.Rd[k] == self.rd[k] * (1 - self.m.exp(self.m.sum(terms))))

        for l in range(self.L):
            terms = [self.m.log(1 - self.qdc[k][l] * self.rdc[k][l] * self.Rd[k] + 1e-6) for k in range(self.K)]
            self.m.Equation(self.Rc[l] == 1 - self.m.exp(self.m.sum(terms)))

    def generate_objective(self):
        """
        Function of generating objective function.
        """
        if self.obj.lower() == 'min':
            self.m.Obj(self.m.sum(self.Rc))
        elif self.obj.lower() == 'max':
            self.m.Obj(-self.m.sum(self.Rc))
        else:
            raise ValueError("Invalid value for 'obj'. "
                             "Expected 'min' or 'max', but got '{}'.".format(self.obj))

    def solve(self):
        """
        Function of solving the reliability optimization based on GEKKO.
        """
        self.m.solve()

