import numpy as np
import scipy as sp
from mary9 import Mary9Optimizer


# second type of objective
def rosenbrock(x, **kwargs):
    if 'sensi_orders' in kwargs:
        sensi_orders = kwargs['sensi_orders']
    else:
        sensi_orders = (0,)

    if 0 in sensi_orders and 1 in sensi_orders:
        return sp.optimize.rosen(x), sp.optimize.rosen_der(x)
    elif 0 in sensi_orders:
        return sp.optimize.rosen(x)

dim_full = 14
lb = -5 * np.ones((dim_full, 1))
ub = 5 * np.ones((dim_full, 1))

optimizer = Mary9Optimizer(objective_function=rosenbrock,
                           lower_bounds=lb,
                           upper_bounds=ub,
                           n_parameters=14)
results = optimizer.minimize()
