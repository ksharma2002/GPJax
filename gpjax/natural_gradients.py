from copy import deepcopy
from typing import Callable, Dict, Tuple

import jax.numpy as jnp
import jax.scipy as jsp
from jax import value_and_grad
from jaxtyping import f64

from .config import get_defaults
from .gps import AbstractPosterior
from .parameters import build_trainables, trainable_params, transform
from .types import Dataset
from .utils import I
from .variational_families import (
    AbstractVariationalFamily,
    ExpectationVariationalGaussian,
    NaturalVariationalGaussian,
)
from .variational_inference import StochasticVI

DEFAULT_JITTER = get_defaults()["jitter"]


def natural_to_expectation(
    natural_moments: Dict, jitter: float = DEFAULT_JITTER
) -> Dict:
    """
    Translate natural parameters to expectation parameters.

    In particular, in terms of the Gaussian mean μ and covariance matrix μ for the Gaussian variational family,

        - the natural parameteristaion is θ = (S⁻¹μ, -S⁻¹/2)
        - the expectation parameters are  η = (μ, S + μ μᵀ).

    This function solves these eqautions in terms of μ and S to convert θ to η.

    Writing θ = (θ₁, θ₂), we have that S⁻¹ = -2θ₂ . Taking the cholesky decomposition of the inverse covariance,
    S⁻¹ = LLᵀ and defining C = L⁻¹, we have S = CᵀC and μ = Sθ₁ = CᵀC θ₁.

    Now from here, using μ and S found from θ, we compute η as η₁ = μ, and  η₂ = S + μ μᵀ.

    Args:
        natural_moments: A dictionary of natural parameters.
        jitter (float): A small value to prevent numerical instability.
    Returns:
        Dict: A dictionary of Gaussian moments under the expectation parameterisation.
    """

    natural_matrix = natural_moments["natural_matrix"]
    natural_vector = natural_moments["natural_vector"]
    m = natural_vector.shape[0]

    # S⁻¹ = -2θ₂
    S_inv = -2 * natural_matrix
    S_inv += I(m) * jitter

    # S⁻¹ = LLᵀ
    L = jnp.linalg.cholesky(S_inv)

    # C = L⁻¹I
    C = jsp.linalg.solve_triangular(L, I(m), lower=True)

    # S = CᵀC
    S = jnp.matmul(C.T, C)

    # μ = Sθ₁
    mu = jnp.matmul(S, natural_vector)

    # η₁ = μ
    expectation_vector = mu

    # η₂ = S + μ μᵀ
    expectation_matrix = S + jnp.matmul(mu, mu.T)

    return {
        "expectation_vector": expectation_vector,
        "expectation_matrix": expectation_matrix,
    }


def _expectation_elbo(
    posterior: AbstractPosterior,
    variational_family: AbstractVariationalFamily,
    train_data: Dataset,
) -> Callable[[Dict, Dataset], float]:
    """
    Construct evidence lower bound (ELBO) for variational Gaussian under the expectation parameterisation.
    Args:
        posterior: An instance of AbstractPosterior.
        variational_family: An instance of AbstractVariationalFamily.
    Returns:
        Callable: A function that computes the ELBO.
    """
    expectation_vartiational_gaussian = ExpectationVariationalGaussian(
        prior=variational_family.prior,
        inducing_inputs=variational_family.inducing_inputs,
    )
    svgp = StochasticVI(
        posterior=posterior, variational_family=expectation_vartiational_gaussian
    )

    return svgp.elbo(train_data, transformations=None, negative=True)


def _stop_gradients_nonmoments(params: Dict) -> Dict:
    """
    Stops gradients for non-moment parameters.
    Args:
        params: A dictionary of parameters.
    Returns:
        Dict: A dictionary of parameters with stopped gradients.
    """
    trainables = build_trainables(params, False)
    moment_trainables = build_trainables(params["variational_family"]["moments"], True)
    trainables["variational_family"]["moments"] = moment_trainables
    params = trainable_params(params, trainables)
    return params


def _stop_gradients_moments(params: Dict) -> Dict:
    """
    Stops gradients for moment parameters.
    Args:
        params: A dictionary of parameters.
    Returns:
        Dict: A dictionary of parameters with stopped gradients.
    """
    trainables = build_trainables(params, True)
    moment_trainables = build_trainables(params["variational_family"]["moments"], False)
    trainables["variational_family"]["moments"] = moment_trainables
    params = trainable_params(params, trainables)
    return params


def natural_gradients(
    stochastic_vi: StochasticVI,
    train_data: Dataset,
    transformations: Dict,
) -> Tuple[Callable[[Dict, Dataset], Dict]]:
    """
    Computes the gradient with respect to the natural parameters. Currently only implemented for the natural variational Gaussian family.
    Args:
        posterior: An instance of AbstractPosterior.
        variational_family: An instance of AbstractVariationalFamily.
        train_data: A Dataset.
        transformations: A dictionary of transformations.
    Returns:
        Tuple[Callable[[Dict, Dataset], Dict]]: Functions that compute natural gradients and hyperparameter gradients respectively.
    """
    posterior = stochastic_vi.posterior
    variational_family = stochastic_vi.variational_family

    # The ELBO under the user chosen parameterisation xi.
    xi_elbo = stochastic_vi.elbo(train_data, transformations, negative=True)

    # The ELBO under the expectation parameterisation, L(η).
    expectation_elbo = _expectation_elbo(posterior, variational_family, train_data)

    if isinstance(variational_family, NaturalVariationalGaussian):

        def nat_grads_fn(params: Dict, trainables: Dict, batch: Dataset) -> Dict:
            """
            Computes the natural gradients of the ELBO.
            Args:
                params: A dictionary of parameters.
                trainables: A dictionary of trainables.
                batch: A Dataset.
            Returns:
                Dict: A dictionary of natural gradients.
            """
            # Transform parameters to constrained space.
            params = transform(params, transformations)

            # Get natural moments θ.
            natural_moments = params["variational_family"]["moments"]

            # Get expectation moments η.
            expectation_moments = natural_to_expectation(natural_moments)

            # Full params with expectation moments.
            expectation_params = deepcopy(params)
            expectation_params["variational_family"]["moments"] = expectation_moments

            # Compute gradient ∂L/∂η:
            def loss_fn(params: Dict, batch: Dataset) -> f64["1"]:
                # Determine hyperparameters that should be trained.
                trains = deepcopy(trainables)
                trains["variational_family"]["moments"] = build_trainables(
                    params["variational_family"]["moments"], True
                )
                params = trainable_params(params, trains)

                # Stop gradients for non-moment parameters.
                params = _stop_gradients_nonmoments(params)

                return expectation_elbo(params, batch)

            value, dL_dexp = value_and_grad(loss_fn)(expectation_params, batch)

            # This is a renaming of the gradient components to match the natural parameterisation pytree.
            nat_grad = dL_dexp
            nat_grad["variational_family"]["moments"] = {
                "natural_vector": dL_dexp["variational_family"]["moments"][
                    "expectation_vector"
                ],
                "natural_matrix": dL_dexp["variational_family"]["moments"][
                    "expectation_matrix"
                ],
            }

            return value, nat_grad

    else:
        raise NotImplementedError

    def hyper_grads_fn(params: Dict, trainables: Dict, batch: Dataset) -> Dict:
        """
        Computes the hyperparameter gradients of the ELBO.
        Args:
            params: A dictionary of parameters.
            trainables: A dictionary of trainables.
            batch: A Dataset.
        Returns:
            Dict: A dictionary of hyperparameter gradients.
        """

        def loss_fn(params: Dict, batch: Dataset) -> f64["1"]:
            # Determine hyperparameters that should be trained.
            params = trainable_params(params, trainables)

            # Stop gradients for the moment parameters.
            params = _stop_gradients_moments(params)

            return xi_elbo(params, batch)

        value, dL_dhyper = value_and_grad(loss_fn)(params, batch)

        return value, dL_dhyper

    return nat_grads_fn, hyper_grads_fn
