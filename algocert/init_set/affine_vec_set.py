from algocert.init_set.init_set import InitSet


class AffineVecSet(InitSet):

    def __init__(self, x, S, b, theta_set):
        super().__init__(x)
        self.S = S
        self.b = b
        self.m, self.n = S.shape
        self.theta_set = theta_set
        self.theta_var = None
        self.theta_thetaT_var = None

    def __str__(self):
        to_string = f'SET({self.x.name}): affine set with ({self.theta_set})'
        return to_string

    def set_theta_vars(self, theta_var, theta_thetaT_var):
        self.theta_var = theta_var
        self.theta_thetaT_var = theta_thetaT_var

    def get_theta_vars(self):
        return self.theta_var, self.theta_thetaT_var

    def get_theta_dim(self):
        return self.n

    def get_output_var_dim(self):
        return self.m
