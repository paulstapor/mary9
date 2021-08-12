import numpy as np
from typing import Sequence, Callable, Dict
import logging
from scipy.optimize import OptimizeResult
logger = logging.getLogger(__name__)

from .initializer import Initializer
from .population import Population
from .global_optimizers import CMAESWrapper, DiffEvolWrapper, InitialRefiner
from .local_optimizers import FidesWrapper
from .util import rescale_to_bounds, unscale_from_bounds


class Mary9Optimizer:

    def __init__(self,
                 objective_function: Callable,
                 n_equivalent_multi_starts: int = 50,
                 local_search_equivalent: int = 500,
                 gradient_eval_equivalent: int = None,
                 max_fevals: int = None,
                 lower_bounds: np.ndarray = None,
                 upper_bounds: np.ndarray = None,
                 n_parameters: int = None,
                 local_solver_tolerance: float = None,
                 random_seed: int = 0,
    ):

        self.objective_function = objective_function
        self.n_equivalent_multi_starts = n_equivalent_multi_starts
        self.local_search_equivalent = local_search_equivalent
        self.gradient_eval_equivalent = gradient_eval_equivalent
        self.max_fevals = max_fevals
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.n_parameters = n_parameters
        self.local_solver_tolerance = local_solver_tolerance
        self.random_seed = random_seed

        if self.gradient_eval_equivalent is None:
            self.gradient_eval_equivalent = self.n_parameters

        self.refinements_run = 0

        # self._check_consistency()

    def minimize(self) -> OptimizeResult:
        # compute sizes of populations
        self._distribute_budget()

        self._schedule_iterations()

        self._initialize_global_optimizers()

        self._initialize_local_optimizers()

        # combine the remaining samples and create a population for initial refinements

        for i_iter in range(self.n_iterations['global']):
            self._perform_global_iteration(iter=i_iter)
            # self._store_global_iteration()

        self._perform_final_search()

        self._finish()
        
    def _distribute_budget(self):
        # compute the overall budget for the optimizer
        refinement_equivalent = 50
        total_budget = self.n_equivalent_multi_starts * \
                       self.gradient_eval_equivalent * \
                       self.local_search_equivalent
        local_search_cost = self.gradient_eval_equivalent * \
                            self.local_search_equivalent
        refinement_cost = self.gradient_eval_equivalent * refinement_equivalent
        # store budgets for global and local searches
        self.budgets = {
            'total': total_budget,
            'global': int(np.round(.4 * total_budget)),
            'local': int(np.round(.4 * total_budget)),
            'refinement': total_budget - int(np.round(.8 * total_budget)),
        }
        self.popsizes = {}
        self.n_iterations = {}

        self.budgets['CMAES'] = int(np.round(.5 * self.budgets['global']))
        self.budgets['DiffEvol'] = \
            int(self.budgets['global'] - np.round(.5 * self.budgets['global']))
        self.n_refinements = \
            int(np.round(self.budgets['refinement'] / refinement_cost))
        # CMAES uses (popsize + 1) * n_iter evaluations at most
        # DiffEvol uses popsize * n_params * (n_iter + 1) evaluations at most,
        # if polish=False is set
        # --> we need to implement a CMAES population which scale with
        # n_parameters and the available number of multistarts, and a DiffEvol
        # population which only scales with the number of multistarts
        # heuristic: 100 - 1000 iterations per global optimizer makes sense
        # (just the number same as for a local optimization)
        lb_popsize_CMAES = np.max([2 * self.n_equivalent_multi_starts,
                                   np.ceil(1.5 * self.n_parameters)])
        lb_iterations_CMAES = 100
        max_iter_CMAES = 1000
        popsize_CMAES = np.floor(self.budgets['CMAES'] / max_iter_CMAES) - 1
        if popsize_CMAES > lb_popsize_CMAES:
            self.popsizes['CMAES'] = int(popsize_CMAES)
            self.n_iterations['CMAES'] = int(max_iter_CMAES)
        else:
            n_iter_CMAES = np.floor(self.budgets['CMAES'] / lb_popsize_CMAES)
            if n_iter_CMAES < lb_iterations_CMAES:
                raise ValueError("No enough multi-starts specified to compute "
                                 "ensure exploration of parameter space")
            self.popsizes['CMAES'] = int(lb_iterations_CMAES)
            self.n_iterations['CMAES'] = int(n_iter_CMAES)


        lb_popsize_DE = 2 * self.n_equivalent_multi_starts
        lb_iterations_DiffEvol = 40
        max_iter_DE = 200
        popsize_DiffEvol = np.floor(self.budgets['DiffEvol'] / max_iter_DE) - 1
        if popsize_DiffEvol > lb_popsize_DE:
            self.popsizes['DiffEvol'] = int(popsize_DiffEvol)
            self.n_iterations['DiffEvol'] = int(max_iter_DE)
        else:
            n_iter_DE = np.floor(self.budgets['DiffEvol'] / lb_popsize_DE)
            if n_iter_DE < lb_iterations_DiffEvol:
                raise ValueError("No enough multi-starts specified to compute "
                                 "ensure exploration of parameter space")
            self.popsizes['DiffEvol'] = int(lb_iterations_DiffEvol)
            self.n_iterations['DiffEvol'] = int(n_iter_DE)

        self.popsizes['initial_refinements'] = \
            int(np.round(.25 * self.n_refinements))
        n_runtime_refinements = self.n_refinements - \
                                self.popsizes['initial_refinements']
        n_runtime_refinements = np.linspace(0, n_runtime_refinements, 4)
        n_runtime_refinements = [int(np.round(nr))
                                 for nr in n_runtime_refinements]
        self.popsizes['runtime_refinements'] = [
            n_runtime_refinements[ir] - n_runtime_refinements[ir - 1]
            for ir in range(1, len(n_runtime_refinements))
        ]

        # Example: say 40 fold MS, 10 params, 500 iters --> total budget = 200k
        # local searches: 20, overall 100k
        # global exploration: 100k
        # budget CMAES: 40k (roughly 8 multi-starts)
        # budget DiffEvol: 40k (roughly 8 multi-starts * 500 iterations * n_par)
        # iterations CMAES: 1000 = 500 * 2
        # popsize CMAES: 40 = 8 * 10 / 2 (MS * n_par)
        # iterations DiffEvol: 200 = 500 / 2.5
        # popsize DiffEvol: 20 = 8 * 2.5
        # budget refinement: 20k
        # cost per refinement: 10(n_parameters) * 50 = 500
        # refinements: 40
        # 8 refinements drawn from additional points
        # 32 refinements during optimization:
        # 200 global iterations in total,
        # we don't want to refine during the first 10% and the last 10%,
        # structure 20 + 160 + 20
        # --> 1 refinement each 5 global iterations
        
        self.n_local_searches = np.round(
            self.budgets['local'] / local_search_cost)

        self.budgets['sampling_CMAES'] = np.max(
            [int(np.round(self.budgets['total'] / 1000)),
             int(np.ceil(2.5 * self.popsizes['CMAES']))]
        )
        self.budgets['sampling_DiffEvol'] = np.max(
            [int(np.round(self.budgets['total'] / 1000)),
             int(2.5 * self.popsizes['DiffEvol'])]
        )
        self.n_iterations['global'] = max([self.n_iterations['CMAES'],
                                           self.n_iterations['DiffEvol']])

    def _initialize_global_optimizers(self):
        """
        Global exploration phase is set up. At the moment, Mary9 uses two
        gradient free global optimizers (CMAES and Differential Evolution) and
        additional "initial refinement", in which remaining parameter vectors
        from initial sampling, which weren't chosen for actual optimization,
        are used to initialize short gradient-based local optimization runs.
        If those deliver good results, the populations of the global
        gradient-free optimizers are updated accordingly.
        """
        initializer = Initializer(
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            objective_function=self.objective_function,
            n_parameters=self.n_parameters
        )

        # sample population for CMAES
        (
            proposal_cmaes,
            full_proposal_cmaes,
            fitness_cmaes,
            selection_info_cmaes
        ) = initializer.sample_initial_points(
            n_samples=self.popsizes['CMAES'],
            random_seed=self.random_seed,
            strategy=(.1, .3, .6, .0),
            sampling_budget=self.budgets['sampling_CMAES']
        )
        # initialize CMAES
        self.cmaes_optimizer = CMAESWrapper(
            population=proposal_cmaes,
            objective_function=self.objective_function,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds
        )

        # sample population for Differential Evolution solver
        # initialize Differential Evolution solver
        (
            proposal_diffevol,
            full_proposal_diffevol,
            fitness_diffevol,
            selection_info_diffevol
        ) = initializer.sample_initial_points(
            n_samples=self.popsizes['DiffEvol'],
            random_seed=self.random_seed,
            strategy=(.1, .2, .2, .5),
            sampling_budget=self.budgets['sampling_DiffEvol']
        )
        # initialize CMAES
        self.diffevol_optimizer = DiffEvolWrapper(
            population=proposal_diffevol,
            objective_function=self.objective_function,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            maxiter=self.n_iterations['DiffEvol']
        )

        # create populations for initial refinement phase. Remaining points
        # from sampling of CMAES and Differential Evolution are used
        # First compute the size of the populations
        popsize_init_refine_cmaes = int(np.round(
            .5 * self.popsizes['initial_refinements']))
        popsize_init_refine_diffevol = int(
            self.popsizes['initial_refinements'] - popsize_init_refine_cmaes)
        # now create the initial refinement populations for both optimizers
        # CMAES
        self.initial_refiner_cmaes = self._setup_initial_refiner(
            full_proposal=full_proposal_cmaes,
            selection_info=selection_info_cmaes,
            popsize=popsize_init_refine_cmaes,
        )
        # Differential Evolution
        self.initial_refiner_diffevol = self._setup_initial_refiner(
            full_proposal=full_proposal_diffevol,
            selection_info=selection_info_diffevol,
            popsize=popsize_init_refine_diffevol,
        )

    def _setup_initial_refiner(
            self,
            full_proposal: Population,
            selection_info: Dict,
            popsize: int
    ):
        # Collect the remaining points (fitness and diversity)
        remaining_fitness = np.array(
            full_proposal.get_fitness())[
            selection_info['remaining']]
        remaining_diversity = np.array(
            full_proposal.get_diversity())[
            selection_info['remaining']]
        # select a mixed sample
        total_ranking = np.argsort(np.argsort(remaining_fitness)) + \
                        np.argsort(np.argsort(remaining_diversity)[::-1])
        proposal_initial_refinements = [
            full_proposal.individuals[ind].x
            for ind in total_ranking[:popsize]
        ]
        fitness_initial_refinements = [
            full_proposal.individuals[ind].fitness
            for ind in total_ranking[:popsize]
        ]
        # create population object from best remaining points
        proposal_initial_refinements = Population(
            population=proposal_initial_refinements,
            fitness=fitness_initial_refinements
        )

        # create mulit-start local optimizer with many start points
        return InitialRefiner(
            population=proposal_initial_refinements,
            objective_function=self.objective_function,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds
        )

    def _schedule_iterations(self):

        calls_to_CMAES = np.linspace(
            0, self.n_iterations['global'], self.n_iterations['CMAES'],
            endpoint=False)
        calls_to_CMAES = [int(iter) for iter in calls_to_CMAES]

        calls_to_DiffEvol = np.linspace(
            0, self.n_iterations['global'], self.n_iterations['DiffEvol'],
            endpoint=False)
        calls_to_DiffEvol = [int(iter) for iter in calls_to_DiffEvol]

        calls_to_InitialRefine = (
            int(np.round(0.1 * self.n_iterations['global'])),)
        calls_to_RuntimeRefine = (
            int(np.round(0.3 * self.n_iterations['global'])),
            int(np.round(0.5 * self.n_iterations['global'])),
            int(np.round(0.7 * self.n_iterations['global'])),
        )

        self.iterations = []
        for iteration in range(self.n_iterations['global']):
            tmp_iteration = []
            if iteration in calls_to_CMAES:
                tmp_iteration.append('CMAES')
            if iteration in calls_to_DiffEvol:
                tmp_iteration.append('DiffEvol')
            if iteration in calls_to_InitialRefine:
                tmp_iteration.append('InitialRefine')
            if iteration in calls_to_RuntimeRefine:
                tmp_iteration.append('RuntimeRefine')

            self.iterations.append(tuple(tmp_iteration))

    def _initialize_local_optimizers(self):

        self.refinement_local_optimizer = FidesWrapper(
            lower_bounds=self.lower_bounds.flatten(),
            upper_bounds=self.upper_bounds.flatten(),
            options={'maxiter': 50},
        )
        self.final_local_optimizer = FidesWrapper(
            lower_bounds=self.lower_bounds.flatten(),
            upper_bounds=self.upper_bounds.flatten(),
            options={'maxiter': 1000},
        )

    def _perform_global_iteration(self, iter):

        if 'CMAES' in self.iterations[iter]:
            next(self.cmaes_optimizer)
            # Now update the populations of other optimization algorithms
            if 'DiffEvol' in self.iterations[iter]:
                n_update_points = int(min([np.round(
                    .05 * self.cmaes_optimizer.population.popsize), 5]))
                self._share_knowledge(
                    update_from=self.cmaes_optimizer.population,
                    update_to=self.diffevol_optimizer.population,
                    update_popsize=n_update_points,
                    update_strategy=(.6, .0, .0, .4),
                    update_threshold='only best'
                )

        if 'DiffEvol' in self.iterations[iter]:
            next(self.diffevol_optimizer)
            # Now update the populations of other optimization algorithms
            n_update_points = int(min([np.round(
                .05 * self.diffevol_optimizer.population.popsize), 5]))
            self._share_knowledge(
                update_from=self.diffevol_optimizer.population,
                update_to=self.cmaes_optimizer.population,
                update_popsize=n_update_points,
                update_strategy=(.6, .0, .0, .4),
                update_threshold='only best'
            )

        if 'InitialRefine' in self.iterations[iter]:
            self.initial_refiner_cmaes.run()
            self._share_knowledge(
                update_from=self.initial_refiner_cmaes.population,
                update_to=self.cmaes_optimizer.population,
                update_popsize=int(
                    np.round(.2 * self.cmaes_optimizer.population.popsize)),
                update_strategy=(.3, .2, .5, .0),
                update_threshold='in between'
            )

            self.initial_refiner_diffevol.run()
            self._share_knowledge(
                update_from=self.initial_refiner_diffevol.population,
                update_to=self.diffevol_optimizer.population,
                update_popsize=int(
                    np.round(.2 * self.diffevol_optimizer.population.popsize)),
                update_strategy=(.15, .15, .0, .7),
                update_threshold='in between'
            )

        if 'RuntimeRefine' in self.iterations[iter]:
            whole_population = Population.merge_populations(
                [self.cmaes_optimizer.population,
                 self.diffevol_optimizer.population]
            )
            refinement_points, selection_info = Initializer.select_subsample(
                proposal=whole_population,
                n_samples=self.popsizes['runtime_refinements'][
                    self.refinements_run],
                strategy=(.2, .2, .5, .1),
                ban_distance=.5 / whole_population.popsize
            )
            for i_point, point in enumerate(refinement_points):
                tmp_id = selection_info['chosen'][i_point]
                if tmp_id >= self.cmaes_optimizer.population.popsize:
                    id = tmp_id - self.cmaes_optimizer.population.popsize
                    pop = self.diffevol_optimizer.population
                else:
                    pop = self.cmaes_optimizer.population
                    id = tmp_id

                individual = pop.individuals[id]
                # run Fides
                # update that point in the population
                x0 = rescale_to_bounds(parameters=individual.x,
                                       lower_bounds=self.lower_bounds,
                                       upper_bounds=self.upper_bounds)
                tmp_result = self.refinement_local_optimizer.minimize(
                    objective_function=self.objective_function,
                    x0=x0,
                    id=str(id)
                )
                individual.fitness = tmp_result.fval
                individual.x = unscale_from_bounds(
                    parameters=tmp_result.x,
                    lower_bounds=self.lower_bounds,
                    upper_bounds=self.upper_bounds
                )

            self.refinements_run += 1

    def _share_knowledge(
            self,
            update_from,
            update_to,
            update_popsize,
            update_strategy,
            update_threshold
    ):
        # select best points from new population
        suggested_points, selection_info = \
            Initializer.select_subsample(
                proposal=update_from,
                n_samples=update_popsize,
                strategy=update_strategy, #(.2, .2, .6, .0),
                ban_distance=2. / update_from.popsize
            )

        # Now select weakest point from old population
        weakest_inds, info_weakest = \
            Initializer.select_weakest_sample(
                proposal=update_to,
                n_samples=update_popsize,
            )

        # Now try to exchange points, if possible
        fittest_old = np.nanmin(update_to.get_fitness())
        for i_vector, _ in enumerate(suggested_points):
            # get the proposed new point
            id_new = selection_info['chosen'][i_vector]
            individual_new = update_from.individuals[id_new]
            # get the old point which should be replaced
            id_old = weakest_inds[i_vector]
            individual_old = update_to.individuals[id_old]

            # only replace if new point is better
            if update_threshold == 'only best':
                threshold = fittest_old
            elif update_threshold == 'always if better':
                threshold = individual_old.fitness
            elif update_threshold == 'in between':
                threshold = .5 * fittest_old + .5 * individual_old.fitness
            else:
                raise Exception('Unknown thresholding for knowledge exchange.')
            if individual_new.fitness < threshold:
                update_to.individuals[id_old] = individual_new

    def _perform_final_search(self):

        whole_population = Population.merge_populations(
            [self.diffevol_optimizer.population,
             self.cmaes_optimizer.population]
        )
        final_search_points, selection_info = Initializer.select_subsample(
            proposal=whole_population,
            n_samples=self.n_local_searches,
            strategy=(.3, .3, .3, .1),
            ban_distance=.1 / whole_population.popsize
        )
        final_results = []
        for id, point in enumerate(final_search_points):
            # run Fides
            point = rescale_to_bounds(point,
                                      lower_bounds=self.lower_bounds,
                                      upper_bounds=self.upper_bounds)
            final_results.append(self.final_local_optimizer.minimize(
                objective_function=self.objective_function,
                x0=point,
                id=str(id)
            ))
