import pytest
import numpy as np
import scipy as sp

from mary9 import Mary9Optimizer


@pytest.fixture
def rosenbrock_objective():
    def objective_function(x, **kwargs):
        if 'sensi_orders' in kwargs:
            sensi_orders = kwargs['sensi_orders']
        else:
            sensi_orders = (0,)

        if 0 in sensi_orders and 1 in sensi_orders:
            return sp.optimize.rosen(x), sp.optimize.rosen_der(x)
        elif 0 in sensi_orders:
            return sp.optimize.rosen(x)

    return objective_function


@pytest.fixture
def bounds():
    dim_full = 14
    lower_bounds = -5 * np.ones((dim_full, 1))
    upper_bounds = 5 * np.ones((dim_full, 1))
    return lower_bounds, upper_bounds


def test_optimizer(bounds, rosenbrock_objective):

    lower_bounds, upper_bounds = bounds

    optimizer = Mary9Optimizer(objective_function=rosenbrock_objective,
                               lower_bounds=lower_bounds,
                               upper_bounds=upper_bounds,
                               n_equivalent_multi_starts=10)
    final_results, final_population = optimizer.minimize()

    final_objective_values = [result.fval for result in final_results]
    final_population_values = [ind.fitness for ind in final_population]
    final_population_values.sort()

    assert np.sum(final_objective_values < 1e-5) > 2
    assert np.sum(final_objective_values < 10) > 10
