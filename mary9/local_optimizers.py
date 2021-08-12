import numpy as np
import fides
from fides.hessian_approximation import BFGS as BFGS
from typing import Dict, Optional, Callable
from scipy.optimize import OptimizeResult


class FidesWrapper:
    """
    Global/Local optimization using the trust region optimizer fides.
    Package Homepage: https://fides-optimizer.readthedocs.io/en/latest
    """

    def __init__(
            self,
            lower_bounds: np.ndarray,
            upper_bounds: np.ndarray,
            hessian_update: Optional['fides.HessianApproximation'] = 'BFGS',
            options: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        options:
            Optimizer options.
        hessian_update:
            Hessian update strategy. If this is None, Hessian (approximation)
            computed by problem.objective will be used.
        """

        if hessian_update == 'Hybrid':
            hessian_update = fides.HybridUpdate()

        if options is None:
            options = {}

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.options = options
        self.hessian_update = hessian_update

    def minimize(
            self,
            objective_function: Callable,
            x0: np.ndarray,
            id: str,
    ) -> OptimizeResult:

        if fides is None:
            raise ImportError(
                "This optimizer requires an installation of fides. You can "
                "install fides via `pip install fides`."
            )

        args = {'mode': 'mode_fun',
                'sensi_orders': (0, 1)}

        opt = fides.Optimizer(
            fun=objective_function,
            funargs=args,
            ub=self.upper_bounds,
            lb=self.lower_bounds,
            hessian_update=BFGS(),
            options=self.options
        )

        try:
            opt.minimize(x0)
            if opt.converged:
                msg = 'Finished Successfully.'
            else:
                msg = 'Failed to converge'
        except RuntimeError as err:
            msg = str(err)

        optimizer_result = OptimizeResult(
            x=opt.x_min, fval=opt.fval_min, grad=opt.grad_min, hess=opt.hess,
            message=msg, exitflag=opt.exitflag
        )

        return optimizer_result

    def is_least_squares(self):
        return False
