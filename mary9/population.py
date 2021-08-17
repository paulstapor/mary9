import numpy as np
from typing import Sequence, List, Callable, Union, Optional
import logging

logger = logging.getLogger(__name__)


class Individual:

    def __init__(self,
                 x: Sequence,
                 fitness: float = np.inf,
                 diversity: float = 0
                 ):
        self.x = x
        self.fitness = fitness
        self.diversity = diversity

    def __iter__(self):
        yield 'x', self.x
        yield 'fitness', self.fitness
        yield 'diversity', self.diversity


class Population:
    def __init__(
            self,
            population: List[np.ndarray],
            fitness: Sequence[float] = None,
            diversity_matrix: np.ndarray = None,
            critical_distance_matrix: np.ndarray = None,
            max_ban_violations: float = None,
            prune_failed: bool = True,
            scale_to_bounds: bool = False,
            objective_function: Optional[Callable] = None
    ):
        if fitness is None:
            if objective_function is None:
                raise ValueError(
                    "If no fitnes values are passed, an objective function is "
                    "needed to compute the fitness values.")

            Population.evaluate_population_fitness(
                population=population,
                objective_function=objective_function
            )

        if diversity_matrix is None:
            diversity_matrix = Population.evaluate_population_diversity(
                population=population
            )

        self.n_parameters = len(population[0])

        if max_ban_violations is None:
            max_ban_violations = int(np.round(.5 * self.n_parameters))
        self.max_ban_violations = max_ban_violations

        if critical_distance_matrix is None:
            critical_distance_matrix = \
                Population.evaluate_critical_ban_distance(
                    population=population,
                    max_ban_violations=max_ban_violations
                )

        assert len(population) == len(fitness) == diversity_matrix.shape[0]
        self.popsize = len(population)
        self.diversity_matrix = diversity_matrix
        self.critical_distance_matrix = critical_distance_matrix
        self.scale_to_bounds = scale_to_bounds

        self.individuals = []
        indices_to_prune = []
        for ind in range(self.popsize):
            # We don't want to add points which are not evaluable
            if prune_failed and not np.isfinite(fitness[ind]):
                indices_to_prune.append(ind)
                continue

            ind_diversity = np.sum(diversity_matrix[ind, :]) / self.popsize
            self.individuals.append(
                Individual(
                    x=population[ind],
                    fitness=fitness[ind],
                    diversity=ind_diversity
                )
            )

        if prune_failed:
            indices_to_keep = np.array([ind for ind in range(self.popsize)
                                        if ind not in indices_to_prune])
            self.critical_distance_matrix = self.critical_distance_matrix[
                indices_to_keep, :
            ]
            self.critical_distance_matrix = self.critical_distance_matrix[
                :, indices_to_keep
            ]
            self.diversity_matrix = self.diversity_matrix[
                indices_to_keep, :
            ]
            self.diversity_matrix = self.diversity_matrix[
                :, indices_to_keep
            ]
            self.popsize = len(self.individuals)

        self.fitness = self.get_fitness()

    def __len__(self):
        return self.popsize

    def __getitem__(self, items):
        if isinstance(items, list):
            return [self.individuals[item] for item in items]
        elif isinstance(items, int):
            return self.individuals[items]

    def get_parameters(self):
        return [ind.x for ind in self.individuals]

    def get_fitness(self):
        return [ind.fitness for ind in self.individuals]

    def set_fitness(self, fitness):
        assert len(fitness) == self.popsize
        for ind, fit in zip(self.individuals, fitness):
            ind.fitness = fit

    def set_parameters(self, parameters):
        assert len(parameters) == self.popsize
        for ind, x in zip(self.individuals, parameters):
            ind.x = x

    def set_single_fitness(self, ind, fitness):
        self.individuals[ind].fitness = fitness

    def set_single_parameters(self, ind, parameters):
        self.individuals[ind].x = parameters

    def get_diversity(self):
        return [ind.diversity for ind in self.individuals]

    def get_diversity_matrix(self):
        return self.diversity_matrix

    def get_critical_distance_matrix(self):
        return self.critical_distance_matrix

    def evaluate_diversity(self):
        self.diversity_matrix = Population.evaluate_population_diversity(
            self.get_parameters()
        )
        for ind in range(self.popsize):
            self.individuals[ind].diversity = np.sum(
                self.diversity_matrix[ind, :]) / self.popsize

    def evaluate_critical_bans(self):
        self.critical_distance_matrix = \
            Population.evaluate_critical_ban_distance(
                population=self.get_parameters(),
                max_ban_violations=self.max_ban_violations
            )

    @staticmethod
    def evaluate_population_fitness(
            population: List[np.ndarray],
            objective_function: Union[Callable, None]
    ) -> Sequence:
        """
        Takes the population and evaluates it based on two criteria:
         - fitness (objective function value)
         - diversity (i.e., averaged mutual distance)
         It ranks them and creates a common ranking, e.g.:

        :return:
            - self.population_ranking
        """

        # consistency check
        fitness_values = []
        for individual in population:
            fitness_values.append(objective_function(individual))

        return np.array(fitness_values).flatten()

    @staticmethod
    def evaluate_population_diversity(
            population: List[np.ndarray],
    ) -> np.ndarray:

        # consistency check
        population_size = len(population)
        if population_size == 0:
            raise ValueError('Empty population, cannot compute diversity.')
        diversity_matrix = np.zeros((population_size,
                                     population_size))

        for i_ind1, individual1 in enumerate(population):
            for i_ind2 in range(i_ind1 + 1, population_size):
                individual2 = population[i_ind2]
                tmp_dist = np.sum(np.abs(individual1 - individual2))
                diversity_matrix[i_ind1, i_ind2] = tmp_dist
                diversity_matrix[i_ind2, i_ind1] = tmp_dist

        return diversity_matrix

    @staticmethod
    def evaluate_critical_ban_distance(
            population: List[np.ndarray],
            max_ban_violations: int = None
    ) -> np.ndarray:

        # consistency check
        population_size = len(population)
        if population_size == 0:
            raise ValueError('Empty population, cannot compute critical '
                             'distance for bans.')

        if max_ban_violations is None:
            n_parameters = population[0].size
            max_ban_violations = int(
                np.round(n_parameters - np.sqrt(n_parameters)))

        critical_distance_matrix = np.zeros((population_size, population_size))

        for i_ind1, individual1 in enumerate(population):
            for i_ind2 in range(i_ind1 + 1, population_size):
                individual2 = population[i_ind2]
                tmp_dist = np.sort(np.abs(individual1 - individual2))
                tmp_ban_dist = tmp_dist[max_ban_violations]
                critical_distance_matrix[i_ind1, i_ind2] = tmp_ban_dist
                critical_distance_matrix[i_ind2, i_ind1] = tmp_ban_dist

        return critical_distance_matrix

    @staticmethod
    def merge_populations(populations: List,
                          scale_to_bounds: bool=False):
        """
        This function merges a set of populations

        :param populations:
        :return:
        """
        individual_list = []
        fitness_list = []
        for pop in populations:
            for individual in pop.individuals:
                individual_list.append(individual.x)
                fitness_list.append(individual.fitness)

        return Population(
            population=individual_list,
            fitness=fitness_list,
            prune_failed=False,
            scale_to_bounds=scale_to_bounds)
