import gekko
import pandas as pd
from gekko import GEKKO
import gurobipy as grb
from gurobipy import nlfunc
import os
import warnings


class InstanceWarning(Warning):
    pass


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
    def __init__(self, instance: str, opt_type: str = 'cost', solver: str = 'Gurobi'):
        """
        Initialization function.

        :param instance: A string representing the name of current instance.
        :type instance: str
        :param opt_type: Optimization problem type (Default: 'multi'; Optional: 'cost', 'reli', 'flex', 'multi')
        :type opt_type: str
        :param solver: The selected solver (Default: 'Gurobi', Option: 'GEKKO', 'Gurobi')
        :type solver: str
        """
        if opt_type not in ('cost', 'reli', 'flex'):
            raise ValueError("Invalid value for 'opt_type'. "
                             "Expected 'cost', 'reli', or 'flex', but got '{}'.".format(self.obj))

        if solver not in ('GEKKO', 'Gurobi'):
            raise ValueError("Invalid value for 'solver'. "
                             "Expected 'GEKKO' or 'Gurobi', but got '{}'.".format(self.obj))

        self.instance = instance
        self.opt_type = opt_type
        self.solver = solver

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

        if solver == 'GEKKO':
            warnings.warn(
                "GEKKO is an open source optimization package with less "
                "solution performance and convergence than Gurobi. ",
                InstanceWarning
            )
            self.model = GEKKO(remote=True)
            self.model.options.MAX_MEMORY = 5

            # Variables
            self.Qsp = [[self.model.Var(lb=0, integer=True) for _ in range(self.J)] for _ in range(self.I)]
            self.Qpd = [[self.model.Var(lb=0, integer=True) for _ in range(self.K)] for _ in range(self.J)]
            self.Qdc = [[self.model.Var(lb=0, integer=True) for _ in range(self.L)] for _ in range(self.K)]
            self.v0 = self.model.Const(0)
            self.v1 = self.model.Const(1)
            self.qdc = [
                [
                    self.model.if3(
                        self.Qdc[k][l] - 1e-6, self.v0, self.v1
                    ) for l in range(self.L)
                ] for k in range(self.K)
            ]
            self.X = [self.model.Var(lb=0, ub=1, integer=True) for _ in range(self.J)]
            self.Y = [self.model.Var(lb=0, ub=1, integer=True) for _ in range(self.K)]

            if opt_type == 'reli':
                self.Rp = [self.model.Var(lb=0, ub=1, integer=False) for _ in range(self.J)]
                self.Rd = [self.model.Var(lb=0, ub=1, integer=False) for _ in range(self.K)]
                self.Rc = [self.model.Var(lb=0, ub=1, integer=False) for _ in range(self.L)]
                self.qsp = [
                    [
                        self.model.if3(
                            self.Qsp[i][j] - 1e-6, self.v0, self.v1
                        ) for j in range(self.J)
                    ] for i in range(self.I)
                ]
                self.qpd = [
                    [
                        self.model.if3(
                            self.Qpd[j][k] - 1e-6, self.v0, self.v1
                        ) for k in range(self.K)
                    ] for j in range(self.J)
                ]

            if opt_type == 'flex':
                self.Dsp = [
                    [
                        self.model.if3(
                            self.fs[i] - self.fsp[i][j], self.v0, self.v1
                        ) for j in range(self.J)
                    ] for i in range(self.I)
                ]
                self.Fsp = [
                    [
                        self.model.Var(
                            lb=0, ub=1, integer=False
                        ) for _ in range(self.J)
                    ] for _ in range(self.I)
                ]
                self.Fp = [self.model.Var(lb=0, ub=1, integer=False) for _ in range(self.J)]

                self.Dpd = [
                    [
                        self.model.if3(
                            self.fpd[j][k] - self.Fp[j], self.v0, self.v1
                        ) for k in range(self.K)
                    ] for j in range(self.J)
                ]
                self.Fpd = [
                    [
                        self.model.Var(
                            lb=0, ub=1, integer=False
                        ) for _ in range(self.K)
                    ] for _ in range(self.J)
                ]
                self.Fd = [self.model.Var(lb=0, ub=1, integer=False) for _ in range(self.K)]

                self.Ddc = [
                    [
                        self.model.if3(
                            self.fdc[k][l] - self.Fd[k], self.v0, self.v1
                        ) for l in range(self.L)
                    ] for k in range(self.K)
                ]
                self.Fdc = [
                    [
                        self.model.Var(
                            lb=0, ub=1, integer=False
                        ) for _ in range(self.L)
                    ] for _ in range(self.K)
                ]
                self.Fc = [self.model.Var(lb=0, ub=1, integer=False) for _ in range(self.L)]

        elif solver == 'Gurobi':
            self.model = grb.Model(name="CostOptimization")

            # Variables
            self.Qsp = self.model.addVars(self.I, self.J, lb=0, vtype=grb.GRB.INTEGER, name='Qsp')
            self.Qpd = self.model.addVars(self.J, self.K, lb=0, vtype=grb.GRB.INTEGER, name='Qpd')
            self.Qdc = self.model.addVars(self.K, self.L, lb=0, vtype=grb.GRB.INTEGER, name='Qdc')
            self.qdc = self.model.addVars(self.K, self.L, vtype=grb.GRB.BINARY, name='qdc')
            self.X = self.model.addVars(self.J, vtype=grb.GRB.BINARY, name='X')
            self.Y = self.model.addVars(self.K, vtype=grb.GRB.BINARY, name='Y')

            if opt_type == 'reli':
                self.qsp = self.model.addVars(self.I, self.J, vtype=grb.GRB.BINARY, name='qsp')
                self.qpd = self.model.addVars(self.J, self.K, vtype=grb.GRB.BINARY, name='qpd')
                self.Rp = self.model.addVars(self.J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Rp')
                self.Rd = self.model.addVars(self.K, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Rd')
                self.Rc = self.model.addVars(self.L, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Rc')

            if opt_type == 'flex':
                self.Dsp = self.model.addVars(self.I, self.J, vtype=grb.GRB.BINARY, name='Dsp')
                self.Dpd = self.model.addVars(self.J, self.K, vtype=grb.GRB.BINARY, name='Dpd')
                self.Ddc = self.model.addVars(self.K, self.L, vtype=grb.GRB.BINARY, name='Ddc')
                self.Fsp = self.model.addVars(self.I, self.J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Fsp')
                self.Fpd = self.model.addVars(self.J, self.K, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Fpd')
                self.Fdc = self.model.addVars(self.K, self.L, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Fdc')
                self.Fp = self.model.addVars(self.J, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Fp')
                self.Fd = self.model.addVars(self.K, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Fd')
                self.Fc = self.model.addVars(self.L, lb=0, ub=1, vtype=grb.GRB.CONTINUOUS, name='Fc')

    def set_equations(self) -> grb.Model | gekko.GEKKO:
        """
        Function of setting up the constraints for the cost optimization model.

        :return: An optimization model.
        :rtype: grb.Model | gekko.GEKKO
        """
        if self.solver == 'GEKKO':
            for i in range(self.I):
                self.model.Equation(sum(self.Qsp[i][j] for j in range(self.J)) <= self.Cs[i])

            for j in range(self.J):
                self.model.Equation(sum(self.Qpd[j][k] for k in range(self.K)) <= self.Cp[j] * self.X[j])

            for k in range(self.K):
                self.model.Equation(sum(self.Qdc[k][l] for l in range(self.L)) <= self.Cd[k] * self.Y[k])

            for l in range(self.L):
                self.model.Equation(sum(self.Qdc[k][l] for k in range(self.K)) >= self.D[l] * self.SL[l])

            for j in range(self.J):
                self.model.Equation(
                    sum(
                        self.Qsp[i][j] for i in range(self.I)
                    ) == sum(
                        self.Qpd[j][k] for k in range(self.K)
                    )
                )
            for k in range(self.K):
                self.model.Equation(
                    sum(
                        self.Qpd[j][k] for j in range(self.J)
                    ) == sum(
                        self.Qdc[k][l] for l in range(self.L)
                    )
                )

            for l in range(self.L):
                self.model.Equation(sum(self.qdc[k][l] for k in range(self.K)) == 1)

            if self.opt_type == 'reli':
                for j in range(self.J):
                    terms = [
                        self.model.log(
                            1 - self.qsp[i][j] * self.rsp[i][j] * self.rs[i] + 1e-6
                        ) for i in range(self.I)
                    ]
                    self.model.Equation(
                        self.Rp[j] == self.rp[j] * (1 - self.model.exp(self.model.sum(terms)))
                    )

                for k in range(self.K):
                    terms = [
                        self.model.log(
                            1 - self.qpd[j][k] * self.rpd[j][k] * self.Rp[j] + 1e-6
                        ) for j in range(self.J)
                    ]
                    self.model.Equation(
                        self.Rd[k] == self.rd[k] * (1 - self.model.exp(self.model.sum(terms)))
                    )

                for l in range(self.L):
                    terms = [
                        self.model.log(
                            1 - self.qdc[k][l] * self.rdc[k][l] * self.Rd[k] + 1e-6
                        ) for k in range(self.K)
                    ]
                    self.model.Equation(
                        self.Rc[l] == 1 - self.model.exp(self.model.sum(terms))
                    )

            if self.opt_type == 'flex':
                for i in range(self.I):
                    for j in range(self.J):
                        self.model.Equation(
                            self.Fsp[i][j] == self.Dsp[i][j] * self.fs[i] + (1 - self.Dsp[i][j]) * self.fsp[i][j]
                        )

                for j in range(self.J):
                    self.model.Equation(
                        self.Fp[j] * sum(
                            self.Qsp[i][j] for i in range(self.I)
                        ) == sum(
                            self.Qsp[i][j] * self.Fsp[i][j] for i in range(self.I)
                        )
                    )

                for j in range(self.J):
                    for k in range(self.K):
                        self.model.Equation(
                            self.Fpd[j][k] == self.Dpd[j][k] * self.Fp[j] + (1 - self.Dpd[j][k]) * self.fpd[j][k]
                        )

                for k in range(self.K):
                    self.model.Equation(
                        self.Fd[k] * sum(
                            self.Qpd[j][k] for j in range(self.J)
                        ) == sum(
                            self.Qpd[j][k] * self.Fpd[j][k] for j in range(self.J)
                        )
                    )

                for k in range(self.K):
                    for l in range(self.L):
                        self.model.Equation(
                            self.Fdc[k][l] == self.Ddc[k][l] * self.Fd[k] + (1 - self.Ddc[k][l]) * self.fdc[k][l]
                        )

                for l in range(self.L):
                    self.model.Equation(
                        self.Fc[l] * sum(
                            self.Qdc[k][l] for k in range(self.K)
                        ) == sum(
                            self.Qdc[k][l] * self.Fdc[k][l] for k in range(self.K)
                        )
                    )

        else:
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
                        self.qdc[k, l], True, self.Qdc[k, l] >= 1e-1,
                        name=f"Indicator_{k}_{l}"
                    )

            for l in range(self.L):
                self.model.addConstr(
                    grb.quicksum(
                        self.qdc[k, l] for k in range(self.K)
                    ) == 1, name=f"Constraint16_{l}"
                )

            if self.opt_type == 'reli':
                for j in range(self.J):
                    expr = nlfunc.exp(
                        grb.quicksum(
                            nlfunc.log(
                                1 - self.qsp[i, j] * self.rsp[i][j] * self.rs[i] + 1e-6
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
                                1 - self.qpd[j, k] * self.rpd[j][k] * self.Rp[j] + 1e-6
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
                                1 - self.qdc[k, l] * self.rdc[k][l] * self.Rd[k] + 1e-6
                            ) for k in range(self.K)
                        )
                    )
                    self.model.addGenConstrNL(
                        self.Rc[l], 1 - expr,
                        name=f"Constraint9_{l}"
                    )

            if self.opt_type == 'flex':
                for i in range(self.L):
                    for j in range(self.J):
                        self.model.addConstr(
                            self.Fsp[i, j] == self.Dsp[i, j] * self.fs[i] + (1 - self.Dsp[i, j]) * self.fsp[i][j],
                            name=f"Constraint10_{i}_{j}"
                        )

                for j in range(self.J):
                    self.model.addConstr(
                        self.Fp[j] * grb.quicksum(
                            self.Qsp[i, j] for i in range(self.I)
                        ) == grb.quicksum(
                            self.Qsp[i, j] * self.Fsp[i, j] for i in range(self.I)
                        ),
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
                        self.Fd[k] * grb.quicksum(
                            self.Qpd[j, k] for j in range(self.J)
                        ) == grb.quicksum(
                            self.Qpd[j, k] * self.Fpd[j, k] for j in range(self.J)
                        ),
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
                        self.Fc[l] * grb.quicksum(
                            self.Qdc[k, l] for k in range(self.K)
                        ) == grb.quicksum(
                            self.Qdc[k, l] * self.Fdc[k, l] for k in range(self.K)
                        ),
                        name=f"Constraint15_{l}"
                    )

        return self.model

    def generate_objective(self, obj: str = 'min') -> grb.Model | gekko.GEKKO:
        """
        Function of generating optimization objective.

        :param obj: Type of objective (Default: 'min'; Optional: 'min', 'max').
        :type obj: str
        :return: An optimization model.
        :rtype: grb.Model | gekko.GEKKO
        """
        if obj.lower() not in ('min', 'max'):
            raise ValueError("Invalid value for 'obj'. "
                             "Expected 'min' or 'max', but got '{}'.".format(self.obj))

        if self.solver == 'GEKKO':
            if self.opt_type == 'cost':
                # Intermediate variable
                fixed_cost_p = self.model.Intermediate(
                    sum(self.FCp[j] * self.X[j] for j in range(self.J))
                )
                fixed_cost_dc = self.model.Intermediate(
                    sum(self.FCd[k] * self.Y[k] for k in range(self.K))
                )

                cost_sp_list = []
                for i in range(self.I):
                    cost_sp_i = self.model.Intermediate(
                        sum(self.Ssp[i][j] * self.Qsp[i][j] for j in range(self.J))
                    )
                    cost_sp_list.append(cost_sp_i)
                cost_sp = self.model.Intermediate(sum(cost_sp_list))

                cost_pd_list = []
                for j in range(self.J):
                    cost_pd_j = self.model.Intermediate(
                        sum(self.Spd[j][k] * self.Qpd[j][k] for k in range(self.K))
                    )
                    cost_pd_list.append(cost_pd_j)
                cost_pd = self.model.Intermediate(sum(cost_pd_list))

                cost_dc_list = []
                for k in range(self.K):
                    cost_dc_k = self.model.Intermediate(
                        sum(self.Sdc[k][l] * self.Qdc[k][l] for l in range(self.L))
                    )
                    cost_dc_list.append(cost_dc_k)

                cost_dc = self.model.Intermediate(sum(cost_dc_list))

                # Objective function
                if obj.lower() == 'min':
                    self.model.Minimize(fixed_cost_p + fixed_cost_dc + cost_sp + cost_pd + cost_dc)
                else:
                    self.model.Maximize(fixed_cost_p + fixed_cost_dc + cost_sp + cost_pd + cost_dc)

            elif self.opt_type == 'reli':
                if obj == 'min':
                    self.model.Minimize(sum(self.Rc))
                else:
                    self.model.Maximize(sum(self.Rc))

            else:
                if obj == 'min':
                    self.model.Minimize(sum(self.Fc))
                else:
                    self.model.Maximize(sum(self.Fc))

        else:
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

        return self.model

    def solve(self) -> grb.Model | gekko.GEKKO:
        """
        Solves the optimization problem using the GEKKO solver.

        :return: An optimization model.
        :rtype: grb.Model | gekko.GEKKO
        """
        if self.solver == 'GEKKO':
            self.model.solve()
        else:
            self.model.optimize()

        return self.model
