import pytest
import numpy as np

from mary9 import Initializer


@pytest.fixture
def bounds():
    lower_bounds = np.array([-3, -2, -1, -5, -6, -7])
    upper_bounds = np.array([2, 3, 4, 5, 2, 0])
    return lower_bounds, upper_bounds


def objective(x):
    theta = [1, 100, 0.2, 200, 5, 50]
    val = (theta[0] - x[0]) ** 2 + theta[1] * (x[1] - x[0]**2) ** 2 + \
          (theta[2] - x[2]) ** 2 + theta[3] * (x[3] - x[2] ** 2) ** 2 + \
          (theta[4] - x[4]) ** 2 + theta[5] * (x[5] - x[4] ** 2) ** 2

    return val


strategies = ('FDbM', 'FDM', 'FDb', 'FD', 'FM', 'F', 'DbM', 'Db', 'DM',
              'D', 'M', (.25, .25, .25, .25), (.1, .0, .0, .9))

ban_distances = (.01, .1)


def test_sampler(bounds):
    lower_bounds, upper_bounds = bounds
    initializer = Initializer(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        objective_function=objective,
        n_parameters=6
    )

    n_samples = 50
    sampling_budget = 200

    for i_seed in range(0, 10, 2):
        for i_strategy in strategies:
            for i_ban_distance in ban_distances:
                final_proposal, proposal, fitness, selection_info = \
                    initializer.sample_initial_points(
                        n_samples=n_samples,
                        sampling_budget=sampling_budget,
                        random_seed=i_seed,
                        strategy=i_strategy,
                        ban_distance=i_ban_distance,
                        max_ban_violations=3
                )

                assert len(final_proposal) == n_samples
