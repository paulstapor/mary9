import pytest
import numpy as np
import warnings
import re

from mary9 import Initializer, Population


@pytest.fixture
def objective():
    def objective_function(x):
        theta = [1, 100, 0.2, 200, 5, 50]
        val = (theta[0] - x[0]) ** 2 + theta[1] * (x[1] - x[0]**2) ** 2 + \
              (theta[2] - x[2]) ** 2 + theta[3] * (x[3] - x[2] ** 2) ** 2 + \
              (theta[4] - x[4]) ** 2 + theta[5] * (x[5] - x[4] ** 2) ** 2
        return val

    return objective_function


@pytest.fixture
def bounds():
    lower_bounds = np.array([-3, -2, -1, -5, -6, -7])
    upper_bounds = np.array([2, 3, 4, 5, 2, 0])
    return lower_bounds, upper_bounds


@pytest.fixture
def strategies():
    return 'FDbM', 'FDM', 'FDb', 'FD', 'FM', 'F', 'DbM', 'Db', 'DM', \
           'D', 'M', (.25, .25, .25, .25), (.1, .0, .0, .9)


@pytest.fixture
def ban_distances():
    return .01, .1


@pytest.fixture
def small_population():

    # this array of coordinates creates a population with strictly
    # monotonically increasing diversity
    xs = [np.array([0., 0.]),
          np.array([0., 0.9]),
          np.array([0.95, 0.]),
          np.array([0., -1.]),
          np.array([-1.05, 0.]),
          np.array([0.95, 0.9]),
          np.array([0.95, -1.]),
          np.array([-1.05, 1.]),
          np.array([-1.15, -1.2]),
          np.array([0., -2.8]),      # should not be replaces, too diverse
          np.array([1., 2.8]),       # should not be replaces, too diverse
          np.array([-3., 2.]),]      # should not be replaced, too diverse

    fvals = [0.,    # should not be replaced, too fit
             7.,    # -> to be kicked out (place -1)
             2.,    # should not be replaced, too fit
             1.,    # should not be replaced, too fit
             3.,    # -> almost to be kicked out (place -5)
             9.,    # -> to be kicked out (place -2)
             6.,    # -> almost to be kicked out (place -3(/-4))
             4.,    # -> almost to be kicked out (place -6)
             8.,    # -> almost to be kicked out (place -4(/-3))
             5.,    # should not be replaced, too diverse
             10.,   # should not be replaced, too diverse
             11.]   # should not be replaced, too diverse

    # total ranking: [F, 17, F,  F,  9, 16, 11,  8, 11,  D, D, D]
    small_pop = Population(population=xs, fitness=fvals)

    return small_pop


def test_select_weakest_subsample(small_population):
    weakest_3, info_3 = Initializer.select_weakest_sample(
        proposal=small_population, n_samples=3)
    assert weakest_3 == [1, 5, 8]

    weakest_4, info_4 = Initializer.select_weakest_sample(
        proposal=small_population, n_samples=4)
    assert weakest_4 == [1, 5, 8, 6]

    weakest_5, info_5 = Initializer.select_weakest_sample(
        proposal=small_population, n_samples=5)
    assert weakest_5 == [1, 5, 8, 6, 4]

    weakest_6, info_6 = Initializer.select_weakest_sample(
        proposal=small_population, n_samples=6)
    assert weakest_6 == [1, 5, 8, 6, 4, 7]


def test_select_subsample(small_population):
    pass


def test_sample_initial_points(objective, bounds, strategies, ban_distances):
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


def test_sample_initial_points_many_bans(objective, bounds, strategies):
    lower_bounds, upper_bounds = bounds
    initializer = Initializer(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        objective_function=objective,
        n_parameters=6
    )

    n_samples = 50
    sampling_budget = 200
    ban_distance = 0.75

    for i_seed in range(0, 10, 2):
        for i_strategy in strategies:
            with pytest.warns(RuntimeWarning) as record:
                final_proposal, proposal, fitness, selection_info = \
                    initializer.sample_initial_points(
                        n_samples=n_samples,
                        sampling_budget=sampling_budget,
                        random_seed=i_seed,
                        strategy=i_strategy,
                        ban_distance=ban_distance,
                        max_ban_violations=3
                    )

                message = record[0].message.args[0]
                num = int((message.split('only ')[1]).split(' samples')[0])
                assert len(final_proposal) == num


def test_sample_initial_points_lhs(bounds, objective):
    lower_bounds, upper_bounds = bounds
    initializer = Initializer(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        objective_function=objective,
        n_parameters=6
    )

    n_samples = 50

    for i_seed in range(0, 10, 2):
        final_proposal, proposal, fitness, selection_info = \
            initializer.sample_initial_points(
                n_samples=n_samples,
                sampling_budget=n_samples,
                random_seed=i_seed,
                strategy='LHS'
            )
        assert len(final_proposal) == n_samples
