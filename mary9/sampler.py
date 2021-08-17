import numpy as np
from typing import Sequence, Callable, List, Tuple, Dict, Union
import logging
from scipy._lib._util import check_random_state
from warnings import warn

from .population import Population
from .util import rescale_to_bounds_ensemble

logger = logging.getLogger(__name__)


class Sampler:

    def __init__(
            self,
            lower_bounds: Sequence[float],
            upper_bounds: Sequence[float],
            objective_function: Callable,
            n_parameters: int = None,
    ):
        """
        The Sampler class aims at creating a good proposal/population of
        parameter vectors (list of ndarray) for global optimization methods
        (i.e., either multi-start local or hybrid global-local methods.)

        :param lower_bounds:
            lower bound for parmaeter sampling
        :param upper_bounds:
            upper bound for parameter sampling
        :param objective_function:
            Objective function to be optimized, needs to be passed if more
            complex sampling strategies than LHS should be used
        :param n_parameters:
            total number of parameters, must match the length of the bounds
        """

        # assert bounds and number of parameters are consistent
        assert len(lower_bounds) == len(upper_bounds)
        if n_parameters is None:
            n_parameters = len(lower_bounds)
        else:
            assert n_parameters == len(lower_bounds)

        # set values
        self.n_parameters = n_parameters
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # objective function
        self.objective_function = objective_function

    def sample_initial_points(
            self,
            n_samples: int,
            sampling_budget: int = None,
            random_seed: int = 0,
            strategy: Union[str, Tuple[float, float, float, float]] = 'FDbM',
            ban_distance: float = None,
            max_ban_violations: int = None
    ) -> Tuple[
        Union[Population, None],
        Union[Population, None],
        np.ndarray,
        Dict
    ]:
        """
        This method creates initial guesses for optimization procedures, for a
        Mary9 Sampler object. It creates a large proposal of parameter
        vectors via latin hypercube sampling and then selects a subsample based
        on a chosen strategy.

        :param n_samples:
            number of parameter vectors to be sampled
        :param sampling_budget:
            Number of allowed objective function evaluations for creating the
            proposal distribution (for LHS strategy, must coincide with
            n_samples)
        :param random_seed:
            random seed to be used
        :param strategy:
            Type of sampling strategy to be used, can be
            LHS for latin hypercube sampling, or any 4-Tuple of floats, which
            sum up to 1. Those four number give the proportions of following
            sampling types when choosing from a larger sample
                * F: Choose fittest samples
                * D: Choose most diverse samples
                * b: Balance diversity between fittest and most diverse samples
                * M: Choose based on a mixed measure
            Also the following strings can be passed, as short hands for
            specific choices:
                * F: Choose fittest only
                * D: Choose the 1 fittest sample, rest based on diversity
                * Db: Choose the 1 fittest sample and a few diverse ones,
                    rest based on the "balancing" idea
                * M: Choose the 1 fittest sample, rest based on mixed metric
                * FD: Choose some fittest, and some diverse
                * FDb: Choose some fittest, and some diverse, some balanced
                * FM: Choose some fittest, and some based on mixed metric
                * FDM: Choose some fittest, and some diverse, some mixed
                * FDbM: Choose a little bit of everything
            (see where ever for documentation, default: FDbM)
        :param ban_distance:
            Different parameter vectors to be accepted should have at least
            this distance (in normalized coordinates)
        :param max_ban_violations:
            Two parameter vectors in a chosen subsample may violate the
            ban_distance for at most max_ban_violations entries in the
            parameter vector. Otherwise, only one of these parameter vectors is
            kept and the other one banned from the proposal.

        :return final_proposal:
            List of parameter vectors (ndarrays)
        :return proposal:
            The full proposal of all generate points
        :return fitness:
            List of fitness (i.e., objective function values) of proposal
        :return diversity_matrix:
            List of diversity values of proposal
        :return selection_info:
            Additional information about the full proposal: Points listed
            with their selection criterion, banned points, remaining points
        """

        # double check sampling budget and number of samples
        if sampling_budget is None:
            if strategy == 'LHS':
                sampling_budget = n_samples
            else:
                sampling_budget = 5 * n_samples
        else:
            assert sampling_budget >= n_samples
        if strategy == 'LHS':
            assert n_samples == sampling_budget

        proposal = Sampler.sample_LHS(
            n_samples=sampling_budget,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            n_parameters=self.n_parameters,
            random_seed=random_seed,
            scale_to_bounds=False)

        # reset population energies
        fitness = Population.evaluate_population_fitness(
            population=rescale_to_bounds_ensemble(
                proposal,
                lower_bounds=self.lower_bounds,
                upper_bounds=self.upper_bounds
            ),
            objective_function=self.objective_function
        )

        # create a population object
        proposal = Population(
            population=proposal,
            fitness=fitness,
            prune_failed=True
        )

        # return, if we only want a latin hypercube sample
        if strategy == 'LHS':
            return (
                proposal,
                proposal,
                fitness,
                {}
            )

        final_proposal, selection_info = self.select_subsample(
            proposal,
            n_samples,
            strategy,
            ban_distance
        )

        final_proposal = Population(
            population=final_proposal,
            fitness=fitness[selection_info['chosen']]
        )

        return (
            final_proposal,
            proposal,
            fitness,
            selection_info
        )

    @staticmethod
    def select_subsample(
            proposal: Population,
            n_samples: int,
            strategy: Union[Tuple, str] = 'FDbM',
            ban_distance: float = None
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Method for selecting a good proposal of points from a larger LHS sample

        :param proposal:
            Proposal distribution drawn via LHS to select points from
        :param n_samples:
            Number f startpoints to be seelcted from larger sample
        :param strategy:
            Type of sampling strategy to be used, can be
            LHS for latin hypercube sampling, or any 4-Tuple of floats, which
            sum up to 1. See sample_initial_points for more documentation.
        :param ban_distance:
            Different parameter vectors to be accepted should have at least
            this distance (in normalized coordinates)

        :return final_proposal:
            List of parameter vectors (ndarrays)
        :return selection_info:
            Additional information about the full proposal: Points listed
            with their selection criterion, banned points, remaining points
        """

        # gives names to the different proportion parts
        (fitness_proportion, diversity_proportion, balanced_proportion,
         mixed_proportion) = Sampler._handle_strategy(strategy=strategy,
                                                          n_samples=n_samples)

        if ban_distance is None:
            ban_distance = 1. / n_samples

        # first, we compute the numbers of samples from the proportions
        n_fittest_samples = int(np.round(fitness_proportion * n_samples))
        n_diverse_samples = \
            int(np.round(diversity_proportion * n_samples))
        n_balanced_samples = \
            int(np.round(balanced_proportion * n_samples))
        n_mixed_samples = n_samples - (n_fittest_samples + n_diverse_samples
                                       + n_balanced_samples)

        # first, we select the fittest samples accoding to fitness_proportion
        fittest_indices, fittest_samples, banned_indices = \
            Sampler._select_fittest_samples(
                proposal=proposal,
                n_fittest_samples=n_fittest_samples,
                ban_distance=ban_distance
            )

        # then we select those samples which should be as diverse as possible
        diverse_indices, diverse_samples, banned_indices = \
            Sampler._select_diverse_samples(
                proposal=proposal,
                fittest_indices=fittest_indices,
                banned_indices=banned_indices,
                n_diverse_samples=n_diverse_samples,
                ban_distance=ban_distance
            )

        balanced_indices, balanced_samples, banned_indices = \
            Sampler._select_balanced_samples(
                proposal=proposal,
                fittest_indices=fittest_indices,
                diverse_indices=diverse_indices,
                banned_indices=banned_indices,
                n_balanced_samples=n_balanced_samples,
                ban_distance=ban_distance
            )

        mixed_indices, mixed_samples, banned_indices = \
            Sampler._select_mixed_samples(
                proposal=proposal,
                fittest_indices=fittest_indices,
                diverse_indices=diverse_indices,
                balanced_indices=balanced_indices,
                banned_indices=banned_indices,
                n_mixed_samples=n_mixed_samples,
                ban_distance=ban_distance
            )

        # compute the remaining samples
        chosen_indices = fittest_indices + diverse_indices + \
            balanced_indices + mixed_indices
        remaining_indices = [ind for ind in range(proposal.popsize)
                             if ind not in chosen_indices + banned_indices]
        final_proposal = [proposal.get_parameters()[ind]
                          for ind in chosen_indices]

        selection_info = {
            'fittest': fittest_indices,
            'diverse': diverse_indices,
            'balanced': balanced_indices,
            'mixed': mixed_indices,
            'chosen': chosen_indices,
            'banned': banned_indices,
            'remaining': remaining_indices,
        }

        return (
            final_proposal,
            selection_info
        )

    @staticmethod
    def select_weakest_sample(
            proposal: Population,
            n_samples: int,
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Method for selecting a poor proposal of points from population.
        When exchanging points between optimizers, our aim will be to replace
        the weakest point from one optimizer with the best points from another
        optimizer. Choosing good points is somewhat straight forward.
        For choosing weak points, this method uses the following logic:
         - Never replace the best 25% of points in terms of fitness
           [stored in the list "dont_touch"]
         - Never replace the most diverse 25% of points
           [stored in the list "dont_touch"]
         - Now create a mixed ranking across all point, and select the weakest
           ones according to this mixed ranking (i.e., poor fitness and
           diversity), and fill the list of weak points one-by-one, omitting
           those point which may not be touched.

        :param proposal:
            Proposal distribution drawn via LHS to select points from
        :param n_samples:
            Number f startpoints to be selected from larger sample

        :return weak_sample:
            List of parameter vectors (ndarrays)
        :return selection_info:
            Additional information about the full proposal: Points listed
            with their selection criterion, banned points, remaining points
        """

        proposal.evaluate_diversity()
        proposal.evaluate_critical_bans()

        popsize = proposal.popsize
        fitness = proposal.get_fitness()
        diversity = proposal.get_diversity()
        sorted_fitness = np.argsort(fitness)
        ranking_fitness = np.argsort(np.argsort(fitness))
        sorted_diversity = np.argsort(diversity)[::-1]
        ranking_diversity = np.argsort(np.argsort(diversity))[::-1]
        ranking_summed = ranking_diversity + ranking_fitness
        inverse_sorted_mixed = np.argsort(ranking_summed)[::-1]

        n_best = int(np.round(.25 * popsize))
        best_fitness = sorted_fitness[:n_best]
        best_diversity = sorted_diversity[:n_best]
        dont_touch = [ind for ind in range(popsize)
                      if ind in best_fitness or ind in best_diversity]

        candidates = []
        for sample in inverse_sorted_mixed:
            if sample not in dont_touch:
                candidates.append(sample)

            if len(candidates) == n_samples:
                return candidates, {}

    @staticmethod
    def _handle_strategy(
            strategy,
            n_samples
    ) -> Tuple[float, float, float, float]:
        """
        Handles the strategy parameter, which can either be passed as string or
        tuple and converts it accordingly.

        :param strategy:
            Type of sampling strategy to be used, can be
            LHS for latin hypercube sampling, or any 4-Tuple of floats, which
            sum up to 1. See sample_initial_points for more documentation.
        :param n_samples:
            Number f startpoints to be seelcted from larger sample

        :return proportions:
            Strategy encoded as 4-tuple of floats
        """
        # check if the return type is known
        if isinstance(strategy, tuple):
            if len(strategy) != 4:
                raise ValueError()
            proportions = strategy
        elif isinstance(strategy, str):
            if strategy == 'LHS':
                proportions = (.0, .0, .0, .0)
            elif strategy == 'FDbM':
                proportions = (.3, .1, .3, .3)
            elif strategy == 'FDM':
                proportions = (.3, .3, .0, .4)
            elif strategy == 'FDb':
                proportions = (.3, .3, .4, .0)
            elif strategy == 'FD':
                proportions = (.5, .5, .0, .0)
            elif strategy == 'FM':
                proportions = (.3, .0, .0, .7)
            elif strategy == 'F':
                proportions = (1., .0, .0, .0)
            elif strategy == 'D':
                proportions = (1 / n_samples, 1. - 1 / n_samples, .0, .0)
            elif strategy == 'Db':
                proportions = (1 / n_samples, .2, .8 - 1 / n_samples, .0)
            elif strategy == 'DbM':
                proportions = (1 / n_samples, .2 - 1 / n_samples, .4, .4)
            elif strategy == 'DM':
                proportions = (1 / n_samples, .4 - 1 / n_samples, .0, .6)
            elif strategy == 'M':
                proportions = (1 / n_samples, 0., 0., 1. - 1 / n_samples)
            else:
                raise ValueError(f"Unknown type of sampling {strategy}. "
                                 f"Please choose among the known types: LHS, "
                                 "FDbM, FDM, FDb, FD, FM, F, Db, D, and M.")
        else:
            raise TypeError(f"Type of sampling must either be string or a"
                            f"4-tuple, but was {type(strategy)}. Stopping.")

        return proportions

    @staticmethod
    def _select_fittest_samples(
            proposal,
            n_fittest_samples,
            ban_distance
    ):
        # prepare variables
        critical_distance_matrix = proposal.get_critical_distance_matrix()
        fittest_indices = []
        remaining_inds = np.array(range(proposal.popsize))
        banned_indices = []
        fitness = np.array(proposal.get_fitness())

        # walk through indices to add or banish them
        for i_sample in range(n_fittest_samples):
            # update remaining indices
            remaining_fitness = fitness[remaining_inds]
            tmp_fittest_ind = np.argmin(remaining_fitness)
            fittest_ind = remaining_inds[tmp_fittest_ind]
            fittest_indices.append(fittest_ind)

            remaining_inds = np.delete(remaining_inds, tmp_fittest_ind)
            # ban other samples which would be too close to this one
            new_banned_indices = np.argwhere(
                critical_distance_matrix[remaining_inds, fittest_ind]
                < ban_distance)
            new_banned_indices = [ind[0] for ind in new_banned_indices
                                  if ind in remaining_inds]
            banned_indices.extend(new_banned_indices)
            remaining_inds = np.delete(remaining_inds,
                                       [np.where(remaining_inds == ind)[0]
                                        for ind in new_banned_indices])

            if remaining_inds.size == 0:
                total_samples = len(fittest_indices)
                warn("No samples left to select from. Returning a popoulation "
                     f"with only {total_samples} samples, as too many samples "
                     "have been banned.", RuntimeWarning)
                break

        return (
            fittest_indices,
            [proposal[ind] for ind in fittest_indices],
            banned_indices
        )

    @staticmethod
    def _select_diverse_samples(
            proposal,
            fittest_indices,
            banned_indices,
            n_diverse_samples,
            ban_distance
    ):
        if n_diverse_samples < 1:
            return [], [], banned_indices
        # check which indices have already been accepted and which ones can
        # still be used
        accepted_inds = np.array(fittest_indices)
        remaining_inds = [
            ind for ind in range(len(proposal))
            if ind not in accepted_inds and ind not in banned_indices
        ]
        diverse_indices = []
        if len(remaining_inds) == 0:
            return diverse_indices, [], banned_indices
        # we need to assemble the diversity matrix
        diversity_matrix = proposal.get_diversity_matrix()
        critical_distance_matrix = proposal.get_critical_distance_matrix()

        # now run over samples and iteratively add the most diverse one
        for i_sample in range(n_diverse_samples):
            # recompute which samples are left
            remaining_matrix = diversity_matrix[:, remaining_inds]
            # compute cumulated distance of points
            tmp_diversity = np.sum(remaining_matrix[accepted_inds, :], axis=0)
            # take the one which is furthest away of all other points
            tmp_most_diverse = np.argmax(tmp_diversity)
            # Beware: The index is computed from a smaller set. Remap index!
            most_diverse = remaining_inds[tmp_most_diverse]
            # Add this index to the ones which should be chosen
            diverse_indices.append(most_diverse)
            accepted_inds = np.concatenate((accepted_inds,
                                            np.array((most_diverse,))))
            # Update the remaining indices
            remaining_inds = np.delete(remaining_inds, tmp_most_diverse)
            # See if some other index should be banned (unlikely, yet...)
            new_banned_indices = np.argwhere(
                critical_distance_matrix[remaining_inds, most_diverse]
                < ban_distance)
            new_banned_indices = [ind[0] for ind in new_banned_indices
                                  if ind in remaining_inds]
            banned_indices.extend(new_banned_indices)
            remaining_inds = np.delete(remaining_inds,
                                       [np.where(remaining_inds == ind)[0]
                                        for ind in new_banned_indices])

            if remaining_inds.size == 0:
                total_samples = len(fittest_indices + diverse_indices)
                warn("No samples left to select from. Returning a popoulation "
                     f"with only {total_samples} samples, as too many samples "
                     "have been banned.", RuntimeWarning)
                break

        diverse_samples = [proposal[ind] for ind in diverse_indices]

        return diverse_indices, diverse_samples, banned_indices

    @staticmethod
    def _select_balanced_samples(
            proposal,
            fittest_indices,
            diverse_indices,
            banned_indices,
            n_balanced_samples,
            ban_distance
    ):
        if n_balanced_samples < 1:
            return [], [], banned_indices
        # check which indices have already been accepted and which ones can
        # still be used
        accepted_inds = np.array(fittest_indices + diverse_indices)
        remaining_inds = [
            ind for ind in range(len(proposal))
            if ind not in accepted_inds and ind not in banned_indices
        ]
        balanced_indices = []
        if len(remaining_inds) == 0:
            return balanced_indices, [], banned_indices
        # we need to assemble the diversity matrix
        diversity_matrix = proposal.get_diversity_matrix()
        critical_distance_matrix = proposal.get_critical_distance_matrix()
        diversity_to_fittest = np.sum(
            diversity_matrix[:, fittest_indices], axis=1)
        remaining_div_to_f = diversity_to_fittest[remaining_inds]
        # we will need the fitness as well
        fitness = np.array(proposal.get_fitness())

        # As a precursor, we compute the median diversity of the fittest and
        # of the most diverse indices. The balanced indices will be in between.
        # Median diversity of fittest samples:
        av_div_fittest = np.median([diversity_to_fittest[ind]
                                    for ind in fittest_indices])
        # Median diversity of most diverse samples:
        av_div_diverse = np.median([diversity_to_fittest[ind]
                                    for ind in diverse_indices])

        # which approximate diversity should the proposed points have?
        start_div = .9 * av_div_fittest + .1 * av_div_diverse
        end_div = .1 * av_div_fittest + .9 * av_div_diverse
        diversity_goals = np.linspace(start_div, end_div, n_balanced_samples,
                                      endpoint=False)

        n_candidates = int(np.round(.1 * len(proposal)))
        for goal in diversity_goals:
            # run over attempted diversities and find the best matching samples
            remaining_div_to_f = diversity_to_fittest[remaining_inds]
            remaining_fitness = fitness[remaining_inds]
            candidates = np.argsort(
                np.abs(remaining_div_to_f - goal))[:n_candidates]
            # now select the fittest one among those
            tmp_candidate = np.argmin(remaining_fitness[candidates])
            tmp_best_balanced = candidates[tmp_candidate]
            # Beware: The index is computed from a smaller set. Remap index!
            best_balanced = remaining_inds[tmp_best_balanced]
            # Add this index to the ones which should be chosen
            balanced_indices.append(best_balanced)
            # Update the remaining indices
            remaining_inds = np.delete(remaining_inds, tmp_best_balanced)
            # See if some other index should be banned (unlikely, yet...)
            new_banned_indices = np.argwhere(
                critical_distance_matrix[remaining_inds, best_balanced]
                < ban_distance)
            new_banned_indices = [ind[0] for ind in new_banned_indices
                                  if ind in remaining_inds]
            banned_indices.extend(new_banned_indices)
            remaining_inds = np.delete(remaining_inds,
                                       [np.where(remaining_inds == ind)[0]
                                        for ind in new_banned_indices])

            if remaining_inds.size == 0:
                total_samples = len(fittest_indices + diverse_indices +
                                    balanced_indices)
                warn("No samples left to select from. Returning a popoulation "
                     f"with only {total_samples} samples, as too many samples "
                     "have been banned.", RuntimeWarning)
                break

        balanced_samples = [proposal[ind] for ind in balanced_indices]

        return balanced_indices, balanced_samples, banned_indices

    @staticmethod
    def _select_mixed_samples(
            proposal,
            fittest_indices,
            diverse_indices,
            balanced_indices,
            banned_indices,
            n_mixed_samples,
            ban_distance
    ):
        if n_mixed_samples < 1:
            return [], [], banned_indices
        # check which indices have already been accepted and which ones can
        # still be used
        accepted_inds = np.array(
            fittest_indices + diverse_indices + balanced_indices)
        remaining_inds = [
            ind for ind in range(len(proposal))
            if ind not in accepted_inds and ind not in banned_indices
        ]
        mixed_indices = []
        if len(remaining_inds) == 0:
            return mixed_indices, [], banned_indices

        # we need to assemble the diversity matrix
        diversity_matrix = proposal.get_diversity_matrix()
        critical_distance_matrix = proposal.get_critical_distance_matrix()
        fitness = np.array(proposal.get_fitness())

        for i_sample in range(n_mixed_samples):
            # recompute which samples are left
            remaining_matrix = diversity_matrix[:, remaining_inds]
            # compute cummulated distance of points
            tmp_diversity = np.sum(remaining_matrix[accepted_inds, :], axis=0)
            # Now rank the samples by diversity...
            ranking_diversity = np.argsort(np.argsort(tmp_diversity))[::-1]
            # ...and rank them by fitness as well
            tmp_fitness = fitness[remaining_inds]
            ranking_fitness = np.argsort(np.argsort(tmp_fitness))
            # Finally, we want a mixture of both rankings:
            ranking_summed = ranking_diversity + ranking_fitness
            # We want to take the best sample according to the new score
            tmp_best_ind_mixed = np.argmin(ranking_summed)
            # Beware: The index is computed from a smaller set. Remap index!
            best_ind_mixed = remaining_inds[tmp_best_ind_mixed]
            # Add this index to the ones which should be chosen
            mixed_indices.append(best_ind_mixed)
            accepted_inds = np.concatenate((accepted_inds,
                                            np.array((best_ind_mixed,))))
            # Update the remaining indices
            remaining_inds = np.delete(remaining_inds, tmp_best_ind_mixed)
            # See if some other index should be banned (unlikely, yet...)
            new_banned_indices = np.argwhere(
                critical_distance_matrix[remaining_inds, best_ind_mixed]
                < ban_distance)
            new_banned_indices = [ind[0] for ind in new_banned_indices
                                  if ind in remaining_inds]
            banned_indices.extend(new_banned_indices)
            remaining_inds = np.delete(remaining_inds,
                                       [np.where(remaining_inds == ind)[0]
                                        for ind in new_banned_indices])

            if remaining_inds.size == 0:
                total_samples = len(fittest_indices + diverse_indices +
                                    balanced_indices + mixed_indices)
                warn("No samples left to select from. Returning a popoulation "
                     f"with only {total_samples} samples, as too many samples "
                     "have been banned.", RuntimeWarning)
                break

        mixed_samples = [proposal[ind] for ind in mixed_indices]

        return mixed_indices, mixed_samples, banned_indices

    @staticmethod
    def sample_LHS(
            n_samples: int,
            lower_bounds: Sequence,
            upper_bounds: Sequence,
            n_parameters: int,
            random_seed: int,
            scale_to_bounds: bool = True,
    ) -> List[np.ndarray]:
        """
        Generates initial points for optimization from via a latin hypercube
        sampling strategy.

        :param n_samples:
            Number f startpoints to be seelcted from larger sample
        :param lower_bounds:
            lower bound for parmaeter sampling
        :param upper_bounds:
            upper bound for parameter sampling
        :param n_parameters:
            total number of parameters, must match the length of the bounds
        :param random_seed:
            random seed to be used
        :param scale_to_bounds:
            Flag indicating whether sampled points should be sampled within the
            unit hypercube [0, 1]^n_samples or scaled to lower and upper bounds

        :return proposal:
            List of parameter vectors
        """
        # set reandom number generator
        rng = check_random_state(random_seed)

        # Each parameter range needs to be sampled uniformly.
        # The scaled parameter range ([0, 1)) needs to be split into
        # `self.num_population_members` segments, each of which has the
        # following size:
        segsize = 1.0 / n_samples

        # Within each segment we sample from a uniform random distribution.
        # We need to do this sampling for each parameter.

        samples = (segsize * rng.uniform(size=(n_samples, n_parameters)))
        # Offset each segment to cover the entire parameter range [0, 1)
        samples += np.linspace(0., 1., n_samples,
                               endpoint=False)[:, np.newaxis]

        # Create an array for population of candidate solutions.
        unscaled_proposal = [np.full((n_parameters,), np.nan)
                             for _ in range(n_samples)]

        # Initialize population of candidate solutions by permutation of the
        # random samples.
        for iPar in range(n_parameters):
            order = rng.permutation(range(n_samples))
            for i_ord, new_ord in enumerate(order):
                unscaled_proposal[i_ord][iPar] = samples[new_ord][iPar]

        if scale_to_bounds:
            return rescale_to_bounds(proposal=unscaled_proposal,
                                     lower_bounds=lower_bounds,
                                     upper_bounds=upper_bounds)
        else:
            return unscaled_proposal


def rescale_to_bounds(proposal: List[np.ndarray],
                      lower_bounds: Sequence[float],
                      upper_bounds: Sequence[float]) -> List[np.ndarray]:
    """
    Rescales a list of parameter vectors to lower and upper bounds.

    :param proposal:
        List of parameter vectors
    :param lower_bounds:
        lower bound for parmaeter sampling
    :param upper_bounds:
        upper bound for parameter sampling

    :return rescaled_proposal:
        List of scaled parameter vectors
    """
    rescaled_proposal = []
    n_parameters = len(lower_bounds)
    for prop in proposal:
        rescaled_proposal.append(
            np.array([lower_bounds[i_par] +
                      (upper_bounds[i_par] - lower_bounds[i_par]) * prop[i_par]
                      for i_par in range(n_parameters)
                      ]).flatten()
        )

    return rescaled_proposal
