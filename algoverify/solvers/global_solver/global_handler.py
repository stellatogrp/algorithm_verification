import gurobipy as gp
import numpy as np

from algoverify.init_set.mult_trajectory import MultTrajectorySets
from algoverify.solvers.global_solver import (
    BOUND_SET_CANON_METHODS,
    BOUND_STEP_CANON_METHODS,
    OBJ_CANON_METHODS,
    SET_CANON_METHODS,
    STEP_CANON_METHODS,
)


class GlobalHandler(object):

    def __init__(self, CP, **kwargs):
        self.CP = CP
        self.K = self.CP.K
        if 'model' in kwargs:
            self.model = kwargs['model']
        else:
            self.model = None
        self.iterate_list = []
        self.param_list = []
        self.mult_traj_param_list = []
        self.iterate_to_id_map = {}
        self.id_to_iterate_map = {}
        self.iterate_to_lower_bound_map = {}
        self.iterate_to_upper_bound_map = {}
        self.iterate_to_gp_var_map = {}
        self.param_to_bound_map = {}
        self.param_to_lower_bound_map = {}
        self.param_to_upper_bound_map = {}
        self.param_to_gp_var_map = {}
        self.param_to_mult_traj_map = {}  # True or False based on how many trajectories
        self.objective = 0

        if 'add_bounds' in kwargs:
            self.add_bounds = kwargs['add_bounds']
        else:
            self.add_bounds = False

        if 'TimeLimit' in kwargs:
            self.TimeLimit = kwargs['TimeLimit']
        else:
            self.TimeLimit = -1  # a 'default' value to flag that we don't want to set it

        if 'minimize' in kwargs:
            self.minimize = kwargs['minimize']
        else:
            self.minimize = False

    def create_gp_model(self):
        # self.model = gp.Model()
        if self.model is None:
            self.model = gp.Model()
            self.model.setParam('NonConvex', 2)
            # self.model.setParam('MIPFocus', 3)
            # self.model.setParam('OptimalityTol', 1e-4)
            # self.model.setParam('FeasibilityTol', 1e-3)
            self.model.setParam('MIPGap', .01)
            if self.TimeLimit > 0:
                self.model.setParam('TimeLimit', self.TimeLimit)

    def create_iterate_id_maps(self):
        """
        creates self.iterate_to_id_map, self.id_to_iterate_map, self.iterate_list

        ****************************************************************
        Consider DR splitting on the problem

        min .5 x^T P x + c^T x
            s.t. Ax + s = b
                 s \in K

        where K is a cartesian product of cones

        i.e. u^{i+1} = (M + I)^{-1}(z^i - q)
             v^{i+1} = Pi(2u^{i+1} - z^i)
             z^{i+1} = z^i + v^{i+1} - u^{i+1}

        ****************************************************************
        self.iterate_list is a list of the OUTPUT iterates of the steps
        self.iterate_list = [u, v, z]

        self.iterate_to_id_map is a mapping from index of self.iterat_list to variable
        self.iterate_to_id_map = {0: u, 1: v, 2: z}

        self.id_to_iterate_map is the reverse mapping
        self.iterate_to_id_map = {u: 0, v: 1, z: 2}
        """
        steps = self.CP.get_algorithm_steps()
        for i, step in enumerate(steps):
            iterate = step.get_output_var()
            self.iterate_to_id_map[iterate] = i
            self.id_to_iterate_map[i] = iterate
            self.iterate_list.append(iterate)

    def create_param_list(self):
        parameters = self.CP.get_parameter_sets()
        for param in parameters:
            if type(param) == MultTrajectorySets:  # only need to extract once
                param_var = param.sets[0].get_iterate()
                self.param_to_mult_traj_map[param_var] = param
                self.mult_traj_param_list.append(param_var)
            else:
                param_var = param.get_iterate()
                self.param_to_mult_traj_map[param_var] = False
            self.param_list.append(param_var)

    def create_iterate_bound_map(self):
        K = self.K
        for init_set in self.CP.get_init_sets():
            if len(self.mult_traj_param_list) > 0:
                iterate = init_set.get_iterate()
                n = iterate.get_dim()
                for param_var in self.mult_traj_param_list:
                    traj_sets = self.param_to_mult_traj_map[param_var]
                    for i, single_traj in enumerate(traj_sets):
                        lb = np.zeros((K + 1, n))
                        ub = np.zeros((K + 1, n))
                        canon_method = BOUND_SET_CANON_METHODS[type(init_set)]
                        l, u = canon_method(init_set, self)
                        # TODO do for all k in canon_iter
                        for k in init_set.canon_iter:
                            lb[k] = l
                            ub[k] = u
                        # lb[0] = l
                        # ub[0] = u
                        self.iterate_to_lower_bound_map[(iterate, param_var, i)] = lb
                        self.iterate_to_upper_bound_map[(iterate, param_var, i)] = ub
            else:
                iterate = init_set.get_iterate()
                n = iterate.get_dim()
                lb = np.zeros((K + 1, n))
                ub = np.zeros((K + 1, n))
                canon_method = BOUND_SET_CANON_METHODS[type(init_set)]
                l, u = canon_method(init_set, self)
                for k in init_set.canon_iter:
                    lb[k] = l
                    ub[k] = u
                # lb[0] = l
                # ub[0] = u
                self.iterate_to_lower_bound_map[iterate] = lb
                self.iterate_to_upper_bound_map[iterate] = ub

        for iterate in self.iterate_list:
            if iterate not in self.iterate_to_lower_bound_map:
                n = iterate.get_dim()
                lb = np.zeros((K + 1, n))
                ub = np.zeros((K + 1, n))
                self.iterate_to_lower_bound_map[iterate] = lb
                self.iterate_to_upper_bound_map[iterate] = ub
            if len(self.mult_traj_param_list) > 0:
                for param_var in self.mult_traj_param_list:
                    if (iterate, param_var, 0) not in self.iterate_to_lower_bound_map:
                        traj_sets = self.param_to_mult_traj_map[param_var]
                        for i, single_traj in enumerate(traj_sets):
                            lb = np.zeros((K + 1, n))
                            ub = np.zeros((K + 1, n))
                            self.iterate_to_lower_bound_map[(iterate, param_var, i)] = lb
                            self.iterate_to_upper_bound_map[(iterate, param_var, i)] = ub

        steps = self.CP.get_algorithm_steps()
        for k in range(1, self.K + 1):
            for step in steps:
                if k >= step.start_canon:
                    canon_method = BOUND_STEP_CANON_METHODS[type(step)]  # change to the correct dictionary
                    canon_method(step, k, self.iterate_to_id_map,
                                self.iterate_to_lower_bound_map, self.iterate_to_upper_bound_map,
                                self.param_to_lower_bound_map, self.param_to_upper_bound_map)

    def create_param_bound_map(self):
        parameters = self.CP.get_parameter_sets()
        for param_set in parameters:
            if type(param_set) == MultTrajectorySets:
                for i, single_param_set in enumerate(param_set):
                    param = single_param_set.get_iterate()
                    canon_method = BOUND_SET_CANON_METHODS[type(single_param_set)]
                    l, u = canon_method(single_param_set, self)
                    self.param_to_lower_bound_map[(param, i)] = l
                    self.param_to_upper_bound_map[(param, i)] = u
            else:
                param = param_set.get_iterate()
                canon_method = BOUND_SET_CANON_METHODS[type(param_set)]
                l, u = canon_method(param_set, self)
                self.param_to_lower_bound_map[param] = l
                self.param_to_upper_bound_map[param] = u

    def create_iterate_gp_var_map(self):
        """
        creates a Gurobi variable for each iterate
        i.e. if we have x_1, \dots, x_N
            each x_i is a vector of length n
            x will have shape (N + 1, n)

        self.iterate_to_gp_var_map is a dictionary

        self.iterate_to_gp_var_map = {u: u_var, v: v_var, z: z_var}
        """
        K = self.K
        for iterate in self.iterate_list:
            n = iterate.get_dim()

            # adds bounds to variables
            if self.add_bounds:
                lb = self.iterate_to_lower_bound_map[iterate]
                ub = self.iterate_to_upper_bound_map[iterate]
            else:
                lb = -gp.GRB.INFINITY * np.ones((K + 1, n))
                ub = gp.GRB.INFINITY * np.ones((K + 1, n))
            # creates the variable
            var = self.model.addMVar((K + 1, n),
                                     name=iterate.get_name(),
                                     ub=ub,
                                     lb=lb)

            self.iterate_to_gp_var_map[iterate] = var

    def create_param_gp_var_map(self):
        """
        param_to_gp_var_map maps parameter to variable

        self.param_to_gp_var_map = {'q': q_var}
        """
        for param in self.param_list:
            m = param.get_dim()
            if self.add_bounds:
                lb = self.param_to_lower_bound_map[param]
                ub = self.param_to_upper_bound_map[param]
            else:
                lb = -gp.GRB.INFINITY * np.ones(m)
                ub = gp.GRB.INFINITY * np.ones(m)
            var = self.model.addMVar(m,
                                     name=param.get_name(),
                                     ub=ub,
                                     lb=lb)
            self.param_to_gp_var_map[param] = var

    def canonicalize_initial_sets(self):
        for init_set in self.CP.get_init_sets():
            for k in init_set.canon_iter:
                canon_method = SET_CANON_METHODS[type(init_set)]
                canon_method(init_set, self.model, self.iterate_to_gp_var_map, k)

    def canonicalize_parameter_sets(self):
        for param_set in self.CP.get_parameter_sets():
            canon_method = SET_CANON_METHODS[type(param_set)]
            canon_method(param_set, self.model, self.param_to_gp_var_map)

    def canonicalize_steps(self):
        steps = self.CP.get_algorithm_steps()
        for k in range(1, self.K + 1):
            for step in steps:
                if k >= step.start_canon:
                    canon_method = STEP_CANON_METHODS[type(step)]
                    canon_method(step, self.model, k,
                                self.iterate_to_gp_var_map, self.param_to_gp_var_map, self.iterate_to_id_map)

    def canonicalize_objective(self):
        # obj = self.CP.objective
        if not isinstance(self.CP.objective, list):
            obj_list = [self.CP.objective]
        else:
            obj_list = self.CP.objective

        gp_obj = 0
        for obj in obj_list:
            obj_canon = OBJ_CANON_METHODS[type(obj)]
            # new_obj, t, y, z = obj_canon(obj, self.model, self.iterate_to_gp_var_map)
            new_obj = obj_canon(obj, self.model, self.iterate_to_gp_var_map)
            gp_obj += new_obj
        self.objective += gp_obj
        # self.t = t
        # self.y = y
        # self.z = z

        if self.minimize:
            self.model.setObjective(gp_obj, gp.GRB.MINIMIZE)
        else:
            self.model.setObjective(gp_obj, gp.GRB.MAXIMIZE)

    def canonicalize(self, **kwargs):
        # create the gurobipy model
        self.create_gp_model()

        # create the variables for each parameter
        self.create_param_list()

        # create mappings from output to iterate
        self.create_iterate_id_maps()
        if self.add_bounds:
            self.create_param_bound_map()
            self.create_iterate_bound_map()
        self.create_iterate_gp_var_map()
        self.create_param_gp_var_map()
        self.canonicalize_initial_sets()
        self.canonicalize_parameter_sets()
        self.canonicalize_steps()
        self.canonicalize_objective()

    def solve(self, **kwargs):
        self.model.optimize()
        # x = self.iterate_list[-1]
        # print(self.iterate_to_gp_var_map[x].X)
        # print(self.t.getValue())
        # print('pos', self.y.X)
        # print('neg', self.z.X)
        # print('w', (self.y + self.z).getValue())
        # w = (self.y + self.z).getValue()
        # t = self.t.getValue()
        # print(np.round(w - np.abs(t), 4))
        out = dict(
            glob_objval=self.model.objVal,
            glob_runtime=self.model.Runtime,
            glob_bestbound=self.model.objBound,
        )
        try:
            out['gap'] = self.model.MIPGap
        except AttributeError:
            out['gap'] = 0
        return out

    def get_iterate_var_map(self):
        return self.iterate_to_gp_var_map

    def get_param_var_map(self):
        return self.param_to_gp_var_map