import numpy as np
from typing import Callable, Sequence

from .population import Population
from .util import (
    rescale_to_bounds,
    rescale_to_bounds_ensemble,
    unscale_from_bounds
)
from .local_optimizers import FidesWrapper

from cma.evolution_strategy import CMAEvolutionStrategy
from scipy.optimize._differentialevolution import (
    DifferentialEvolutionSolver,
    _status_message
)


class CMAESWrapper:
    def __init__(
            self,
            population: Population,
            objective_function: Callable,
            lower_bounds: Sequence,
            upper_bounds: Sequence):
        self.population = population
        self.strategy = CMAEvolutionStrategy(
            sigma0=0.25,
            x0=0.5 * np.ones((len(lower_bounds), )),
            inopts={'popsize': population.popsize}
        )
        self.objective_function = GlobalOptimizerObjective(
            objective_function,
            lower_bounds,
            upper_bounds
        )

        # This call is needed for initialization
        self.strategy.ask()

    def __next__(self):
        # let CMAES prepare the next step
        self.strategy.tell(
            np.array(self.population.get_parameters()),
            self.population.get_fitness()
        )
        # get solutions candidates from CMAES
        self.population.set_parameters(self.strategy.ask())
        for ix, x in enumerate(self.population.get_parameters()):
            self.population.set_single_fitness(ix, self.objective_function(x))


class DiffEvolWrapper:
    def __init__(
            self,
            population: Population,
            objective_function: Callable,
            lower_bounds: Sequence,
            upper_bounds: Sequence,
            maxiter: int,
    ):
        self.population = population
        self.objective_function = GlobalOptimizerObjective(
            objective_function,
            lower_bounds,
            upper_bounds
        )
        self.solver = DifferentialEvolutionSolver(
            self.objective_function,
            bounds=[(0., 1.) for _ in range(population.n_parameters)],
            strategy='best1bin',
            maxiter=maxiter,
            popsize=population.popsize,
            tol=1e-3,
            polish=False,
            init=np.array(population.get_parameters())
        )

        self.solver.feasible = np.array([True] * population.popsize)
        self.solver.population_energies = np.array(
            self.population.get_fitness())
        self.solver._promote_lowest_energy()

    def __next__(self):
        next(self.solver)
        self.population.set_parameters(list(self.solver.population))
        self.population.set_fitness(list(self.solver.population_energies))


class InitialRefiner:
    def __init__(
            self,
            population: Population,
            objective_function: Callable,
            lower_bounds: np.ndarray,
            upper_bounds: np.ndarray):

        self.population = population
        self.objective_function = objective_function
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.local_optimizer = FidesWrapper(
            lower_bounds=lower_bounds.flatten(),
            upper_bounds=upper_bounds.flatten(),
            options={'maxiter': 15},
        )
        self.results = []

    def minimize(self,
                 *args,
                 **kwargs):
        return self.local_optimizer.minimize(*args, **kwargs)

    def run(self):
        # scale population if necessary, as local optimization is carried out
        # within original bounds
        if self.population.scale_to_bounds:
            tmp_proposal = self.population.get_parameters()
        else:
            tmp_proposal = \
                rescale_to_bounds_ensemble(
                    self.population.get_parameters(),
                    lower_bounds=self.lower_bounds,
                    upper_bounds=self.upper_bounds
                )

        # loop over points in ensemble and run short local optimizations
        for id, ind in enumerate(tmp_proposal):
            tmp_result = self.minimize(
                    objective_function=self.objective_function,
                    x0=ind,
                    id=str(id)
            )
            # update the population object from the local optimization results
            # rescale to unit hypercube for global optimization
            self.population.individuals[id].x = unscale_from_bounds(
                tmp_result.x,
                lower_bounds=self.lower_bounds,
                upper_bounds=self.upper_bounds
            )
            self.population.individuals[id].fitness = tmp_result.fval
            # diversity and bans need to be reevaluated
            self.population.evaluate_diversity()
            self.population.evaluate_critical_bans()
            # TODO: assess memory impact of storing all result objects,
            #  as done here. Actually, this storing should only happen if
            #  something like a debug mode is enabled
            self.results.append(tmp_result)


class GlobalOptimizerObjective:
    def __init__(
            self,
            objective_function: Callable,
            lower_bounds: Sequence,
            upper_bounds: Sequence
    ):
        self.raw_objective_function = objective_function
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)

    def __call__(self, x, *args, **kwargs):
        x_scaled = rescale_to_bounds(x, self.lower_bounds, self.upper_bounds)
        return self.raw_objective_function(x_scaled)
