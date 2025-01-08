import pandas as pd
import gurobipy as grb
from gurobipy import nlfunc
import os


class SupplyChainSolver:
    """
    A class for solving the supply chain optimization problem.

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
        rs (list): List of reliability indexes for each supplier.
        rp (list): List of reliability indexes for each plant.
        rd (list): List of reliability indexes for each DC.
        fs (list): List of flexibility indexes for each supplier.
        FCp (list): List of fixed cost for each plant.
        FCd (list): List of fixed cost for each DC.
        Ssp (list of lists): Matrix of shipping cost from suppliers to plants.
        Spd (list of lists): Matrix of shipping cost from plants to DCs.
        Sdc (list of lists): Matrix of shipping cost from DCs to CZs.
        rsp (list of lists): Matrix of reliability of arcs from suppliers to plants.
        rpd (list of lists): Matrix of reliability of arcs from plants to DCs.
        rdc (list of lists): Matrix of reliability of arcs from DCs to CZs.
        fsp (list of lists): Matrix of volume flexibility of links from suppliers to plants.
        fpd (list of lists): Matrix of volume flexibility of links from plants to DCs.
        fdc (list of lists): Matrix of volume flexibility of links from DCs to CZs.
        Qsp (list of lists): Decision variables for quantities shipped from suppliers to plants.
        Qpd (list of lists): Decision variables for quantities shipped from plants to distribution centers.
        Qdc (list of lists): Decision variables for quantities shipped from DCs to customer zones.
        qsp (list of lists): Binary decision variables indicating whether a supplier is active for a plant.
        qpd (list of lists): Binary decision variables indicating whether a plant is active for a DC.
        qdc (list of lists): Binary decision variables indicating whether a DC is active for a CZ.
        X (list): Binary decision variables for activating plants.
        Y (list): Binary decision variables for activating distribution centers.
    """
    def __init__(self, instance: str, opt_type: str = 'cost'):
        """
        Initialization function.

        :param instance: A string representing the name of current instance.
        :type instance: str
        :param opt_type: Optimization problem type (Default: 'multi'; Optional: 'cost', 'reli', 'flex', 'multi')
        :type opt_type: str
        """
        if opt_type not in ('cost', 'reli', 'flex'):
            raise ValueError("Invalid value for 'opt_type'. "
                             "Expected 'cost', 'reli', or 'flex', but got '{}'.".format(self.obj))

        self.instance = instance
        self.opt_type = opt_type

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

        # Fixed cost
        self.FCp = df_faci_params['pla_cost'].dropna().to_list()
        self.FCd = df_faci_params['dc_cost'].dropna().to_list()

        # Shipping cost
        self.Ssp = [[float(x.split(',')[0]) for x in df_stage1.iloc[i]] for i in range(self.I)]
        self.Spd = [[float(x.split(',')[0]) for x in df_stage2.iloc[j]] for j in range(self.J)]
        self.Sdc = [[float(x.split(',')[0]) for x in df_stage3.iloc[k]] for k in range(self.K)]

        self.model = grb.Model(name="CostOptimization")
        self.model.setParam(grb.GRB.Param.OutputFlag, 0)

        # Variables
        self.Qsp = self.model.addVars(self.I, self.J, lb=0, vtype=grb.GRB.INTEGER, name='Qsp')
        self.Qpd = self.model.addVars(self.J, self.K, lb=0, vtype=grb.GRB.INTEGER, name='Qpd')
        self.Qdc = self.model.addVars(self.K, self.L, lb=0, vtype=grb.GRB.INTEGER, name='Qdc')
        self.qdc = self.model.addVars(self.K, self.L, vtype=grb.GRB.BINARY, name='qdc')
        self.X = self.model.addVars(self.J, vtype=grb.GRB.BINARY, name='X')
        self.Y = self.model.addVars(self.K, vtype=grb.GRB.BINARY, name='Y')

        if opt_type == 'reli':
            self.rs = df_faci_params['sup_reli'].to_list()  # Reliability index of suppliers.
            self.rp = df_faci_params['pla_reli'].dropna().to_list()  # Reliability index of plants.
            self.rd = df_faci_params['dc_reli'].dropna().to_list()  # Reliability index of distribution centers.

            # Reliability of arc
            self.rsp = [[float(x.split(',')[1]) for x in df_stage1.iloc[i]] for i in range(self.I)]
            self.rpd = [[float(x.split(',')[1]) for x in df_stage2.iloc[j]] for j in range(self.J)]
            self.rdc = [[float(x.split(',')[1]) for x in df_stage3.iloc[k]] for k in range(self.K)]

            # Variables
            self.qsp = self.model.addVars(self.I, self.J, vtype=grb.GRB.BINARY, name='qsp')
            self.qpd = self.model.addVars(self.J, self.K, vtype=grb.GRB.BINARY, name='qpd')
            self.Rp = self.model.addVars(self.J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Rp')
            self.Rd = self.model.addVars(self.K, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Rd')
            self.Rc = self.model.addVars(self.L, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Rc')

        if opt_type == 'flex':
            self.fs = df_faci_params['sup_flex'].to_list()  # Flexibility index of suppliers.

            # Volume flexibility of link
            self.fsp = [[float(x.split(',')[2]) for x in df_stage1.iloc[i]] for i in range(self.I)]
            self.fpd = [[float(x.split(',')[2]) for x in df_stage2.iloc[j]] for j in range(self.J)]
            self.fdc = [[float(x.split(',')[2]) for x in df_stage3.iloc[k]] for k in range(self.K)]

            self.Dsp = self.model.addVars(self.I, self.J, vtype=grb.GRB.BINARY, name='Dsp')
            self.Dpd = self.model.addVars(self.J, self.K, vtype=grb.GRB.BINARY, name='Dpd')
            self.Ddc = self.model.addVars(self.K, self.L, vtype=grb.GRB.BINARY, name='Ddc')
            self.Fsp = self.model.addVars(self.I, self.J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Fsp')
            self.Fpd = self.model.addVars(self.J, self.K, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Fpd')
            self.Fdc = self.model.addVars(self.K, self.L, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Fdc')
            self.Fp = self.model.addVars(self.J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Fp')
            self.Fd = self.model.addVars(self.K, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Fd')
            self.Fc = self.model.addVars(self.L, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Fc')
            self.Fpz = self.model.addVars(self.J, vtype=grb.GRB.BINARY, name='Fpz')
            self.Fdz = self.model.addVars(self.K, vtype=grb.GRB.BINARY, name='Fdz')
            self.Fcz = self.model.addVars(self.L, vtype=grb.GRB.BINARY, name='Fcz')

    def set_equations(self):
        """
        Function of setting up the constraints for the cost optimization model.
        """
        for i in range(self.I):
            self.model.addConstr(
                grb.quicksum(
                    self.Qsp[i, j] for j in range(self.J)
                ) <= self.Cs[i],
                name=f"Constraint1_{i}"
            )

        for j in range(self.J):
            self.model.addConstr(
                grb.quicksum(
                    self.Qpd[j, k] for k in range(self.K)
                ) <= self.Cp[j] * self.X[j],
                name=f"Constraint2_{j}"
            )

        for k in range(self.K):
            self.model.addConstr(
                grb.quicksum(
                    self.Qdc[k, l] for l in range(self.L)
                ) <= self.Cd[k] * self.Y[k],
                name=f"Constraint3_{k}"
            )

        for l in range(self.L):
            self.model.addConstr(
                grb.quicksum(
                    self.Qdc[k, l] for k in range(self.K)
                ) >= self.D[l] * self.SL[l],
                name=f"Constraint4_{l}"
            )

        for j in range(self.J):
            self.model.addConstr(
                grb.quicksum(
                    self.Qsp[i, j] for i in range(self.I)
                ) == grb.quicksum(
                    self.Qpd[j, k] for k in range(self.K)
                ),
                name=f"Constraint5_{j}"
            )

        for k in range(self.K):
            self.model.addConstr(
                grb.quicksum(
                    self.Qpd[j, k] for j in range(self.J)
                ) == grb.quicksum(
                    self.Qdc[k, l] for l in range(self.L)
                ),
                name=f"Constraint6_{k}"
            )

        for k in range(self.K):
            for l in range(self.L):
                self.model.addGenConstrIndicator(
                    self.qdc[k, l], True, self.Qdc[k, l] >= 0,
                    name=f"Indicator_qdc_{k}_{l}"
                )

        for l in range(self.L):
            self.model.addConstr(
                grb.quicksum(
                    self.qdc[k, l] for k in range(self.K)
                ) == 1, name=f"Constraint16_{l}"
            )

        if self.opt_type == 'reli':
            for i in range(self.I):
                for j in range(self.J):
                    self.model.addGenConstrIndicator(
                        self.qsp[i, j], True, self.Qsp[i, j] >= 0,
                        name=f"Indicator_qsp_{i}_{j}"
                    )

            for j in range(self.J):
                for k in range(self.K):
                    self.model.addGenConstrIndicator(
                        self.qpd[j, k], True, self.Qpd[j, k] >= 0,
                        name=f"Indicator_qpd_{j}_{k}"
                    )

            for j in range(self.J):
                expr = nlfunc.exp(
                    grb.quicksum(
                        nlfunc.log(
                            1 - self.qsp[i, j] * self.rsp[i][j] * self.rs[i]
                        ) for i in range(self.I)
                    )
                )
                self.model.addGenConstrNL(
                    self.Rp[j], self.rp[j] * (1 - expr),
                    name=f"Constraint7_{j}"
                )

            for k in range(self.K):
                expr = nlfunc.exp(
                    grb.quicksum(
                        nlfunc.log(
                            1 - self.qpd[j, k] * self.rpd[j][k] * self.Rp[j]
                        ) for j in range(self.J)
                    )
                )
                self.model.addGenConstrNL(
                    self.Rd[k], self.rd[k] * (1 - expr),
                    name=f"Constraint8_{k}"
                )

            for l in range(self.L):
                expr = nlfunc.exp(
                    grb.quicksum(
                        nlfunc.log(
                            1 - self.qdc[k, l] * self.rdc[k][l] * self.Rd[k]
                        ) for k in range(self.K)
                    )
                )
                self.model.addGenConstrNL(
                    self.Rc[l], 1 - expr,
                    name=f"Constraint9_{l}"
                )

        if self.opt_type == 'flex':

            for i in range(self.I):
                for j in range(self.J):
                    self.model.addGenConstrIndicator(
                        self.Dsp[i, j], True, self.fs[i], grb.GRB.LESS_EQUAL, self.fsp[i][j],
                        name=f"Constraint_Dsp_{i}_{j}"
                    )

            for j in range(self.J):
                for k in range(self.K):
                    self.model.addGenConstrIndicator(
                        self.Dpd[j, k], True, self.Fp[j], grb.GRB.LESS_EQUAL, self.fpd[j][k],
                        name=f"Constraint_Dpd_{j}_{k}"
                    )

            for k in range(self.K):
                for l in range(self.L):
                    self.model.addGenConstrIndicator(
                        self.Ddc[k, l], True, self.Fd[k], grb.GRB.LESS_EQUAL, self.fdc[k][l],
                        name=f"Constraint_Ddc_{k}_{l}"
                    )

            for i in range(self.I):
                for j in range(self.J):
                    self.model.addConstr(
                        self.Fsp[i, j] == self.Dsp[i, j] * self.fs[i] + (1 - self.Dsp[i, j]) * self.fsp[i][j],
                        name=f"Constraint10_{i}_{j}"
                    )

            for j in range(self.J):
                self.model.addConstr(
                    self.Fp[j] == grb.quicksum(
                        self.Qsp[i, j] * self.Fsp[i, j] for i in range(self.I)
                    ) / (grb.quicksum(
                        self.Qsp[i, j] for i in range(self.I)
                    ) + 1),
                    name=f"Constraint11_{j}"
                )

            for j in range(self.J):
                for k in range(self.K):
                    self.model.addConstr(
                        self.Fpd[j, k] == self.Dpd[j, k] * self.Fp[j] + (1 - self.Dpd[j, k]) * self.fpd[j][k],
                        name=f"Constraint12_{j}_{k}"
                    )

            for k in range(self.K):
                self.model.addConstr(
                    self.Fd[k] == grb.quicksum(
                        self.Qpd[j, k] * self.Fpd[j, k] for j in range(self.J)
                    ) / (grb.quicksum(
                        self.Qpd[j, k] for j in range(self.J)
                    ) + 1),
                    name=f"Constraint13_{k}"
                )

            for k in range(self.K):
                for l in range(self.L):
                    self.model.addConstr(
                        self.Fdc[k, l] == self.Ddc[k, l] * self.Fd[k] + (1 - self.Ddc[k, l]) * self.fdc[k][l],
                        name=f"Constraint14_{k}_{l}"
                    )

            for l in range(self.L):
                self.model.addConstr(
                    self.Fc[l] == grb.quicksum(
                        self.Qdc[k, l] * self.Fdc[k, l] for k in range(self.K)
                    ) / (grb.quicksum(
                        self.Qdc[k, l] for k in range(self.K)
                    ) + 1),
                    name=f"Constraint15_{l}"
                )

        return self

    def generate_objective(self, obj: str = 'min'):
        """
        Function of generating optimization objective.

        :param obj: Type of objective (Default: 'min'; Optional: 'min', 'max').
        :type obj: str
        """
        if obj.lower() not in ('min', 'max'):
            raise ValueError("Invalid value for 'obj'. "
                             "Expected 'min' or 'max', but got '{}'.".format(self.obj))

        if self.opt_type == 'cost':
            if obj.lower() == 'min':
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
                    ),
                    grb.GRB.MINIMIZE
                )
            else:
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
                    ),
                    grb.GRB.MAXIMIZE
                )

        elif self.opt_type == 'reli':
            if obj == 'min':
                self.model.setObjective(
                    grb.quicksum(self.Rc) / self.L,
                    grb.GRB.MINIMIZE
                )
            else:
                self.model.setObjective(
                    grb.quicksum(self.Rc) / self.L,
                    grb.GRB.MAXIMIZE
                )

        else:
            self.model.setParam(grb.GRB.Param.MIPGap, 0.05)

            # if self.instance[0] == 'S':
            #     self.model.setParam(grb.GRB.Param.TimeLimit, 120)
            if self.instance[0] == 'M':
                self.model.setParam(grb.GRB.Param.TimeLimit, 300)
            elif self.instance[0] == 'L':
                self.model.setParam(grb.GRB.Param.TimeLimit, 600)
            else:
                self.model.setParam(grb.GRB.Param.TimeLimit, 1200)

            if obj == 'min':
                self.model.setObjective(
                    grb.quicksum(self.Fc) / self.L,
                    grb.GRB.MINIMIZE
                )
            else:
                self.model.setObjective(
                    grb.quicksum(self.Fc) / self.L,
                    grb.GRB.MAXIMIZE
                )

        return self

    def solve(self):
        """
        Solves the optimization problem using the GEKKO solver.
        """
        self.model.optimize()
        return self.model
