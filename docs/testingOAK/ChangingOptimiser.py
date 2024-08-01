import jax
# Enable Float64 for more stable matrix inversions.
jax.config.update("jax_enable_x64", True)

from dataclasses import dataclass
import warnings
from typing import List, Union
import pandas as pd
import cola

import optax as ox
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Array,
    Float,
    install_import_hook,
    Num,
)
import tensorflow_probability.substrates.jax.bijectors as tfb

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
from gpjax.typing import (
    Array,
    ScalarFloat,
)
from gpjax.distributions import GaussianDistribution
from gpjax.kernels import AdditiveKernel

import matplotlib.pyplot as plt
from matplotlib import rcParams
# plt.style.use(
#     "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
# )
# colors = rcParams["axes.prop_cycle"].by_key()["color"]
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
mpl.rcParams.update(mpl.rcParamsDefault)

#imports AdditiveConjugatePosterior
# from OAK import AdditiveConjugatePosterior

from pprint import pprint
import numpy as np
import pickle

# Things we need:
# list of functions 
# number of points (drawn normal or uniform (if so what range))
# noise level
# training and test split (default 80/20)
# number of points to plot with in each dimension and min and max values 
# to scale data or not to scale Data (default not?), need to specify for both y and X
# kernel (default RBF), can be orthogonal kernel
# trainable noise for likelihood
# maximum interaction depth 
# optimiser (default Adam), can be SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam, AMSGrad, RAdam, Yogi, SLSQP, L-BFGS-B, Powell, COBYLA, trust-constr, Newton-CG, trust-ncg
# number of iterations
# optimiser learning rate

class AdditiveConjugatePosterior(gpx.gps.ConjugatePosterior):
    r"""        
    Build an additive posterior from an additive kernel and a Gaussian likelihood. We have included an
    additional method to allow predictions for specific additive components, as specified by a
    component_list, e.g. [0, 1] corresponds to the second order interaction between zeroth and first inputs.
    """

    def __post__init__(self):
        assert isinstance(self.prior.kernel, AdditiveKernel), "AdditiveConjugatePosterior requires an AdditiveKernel"

    def predict_additive_component(
        self,
        test_inputs: Num[Array, "N D"],
        train_data: gpx.Dataset,
        component_list: List[List[int]]
    ) -> GaussianDistribution:
        r"""Get the posterior predictive distribution for a specific additive component."""
        specific_kernel = self.prior.kernel.get_specific_kernel(component_list)
        return self.predict(test_inputs, train_data, kernel_with_test = specific_kernel)

    def get_sobol_index(self, train_data: gpx.Dataset, component_list: List[int]) -> ScalarFloat:
        """ Return the sobol index for the additive component corresponding to component_list. """
        component_posterior = self.predict_additive_component(train_data.X, train_data, component_list)
        full_posterior= self.predict(train_data.X, train_data) # wasteful as only need means
        return jnp.var(component_posterior.mean()) / jnp.var(full_posterior.mean())

@dataclass()
class OrthogonalRBF(gpx.kernels.AbstractKernel):
    r"""todo only for unit gaussian input measure and zero mean."""
    name: str = "OrthogonalRBF"
    lengthscale: Union[ScalarFloat, Float[Array, " D"]] = gpx.param_field(
        jnp.array(1.0), bijector=tfb.Softplus()
    )

    def __post_init__(self):
        warnings.warn("This kernel is only valid for unit gaussian input measures and zero mean functions.")

    def __call__(self, x: Num[Array, " D"], y: Num[Array, " D"]) -> ScalarFloat:
        r"""Compute an orthogonal RBF kernel between a pair of arrays."""
        x = self.slice_input(x) # [d]
        y = self.slice_input(y) # [d]
        ks = jnp.exp(-0.5 * ((x - y) / self.lengthscale) ** 2) # [d]
        ks -=  self._cov_x_s(x) * self._cov_x_s(y) / self._var_s() # [d]
        return jnp.prod(ks)
    
    def _cov_x_s(self,x):
        l2 = self.lengthscale ** 2
        return jnp.sqrt(l2 / (l2 + 1.0)) * jnp.exp(-0.5 * (x ** 2) / (l2 + 1.0)) # [d]
        
    def _var_s(self):
        return  jnp.sqrt(self.lengthscale ** 2 / (self.lengthscale ** 2 + 2.0)) # [d]

def plot_posterior_with_components(opt_posterior, feature_dimension, xplot1d, Xtr, ytr, sobol=False):
    fig, ax = plt.subplots(ncols=1, nrows=feature_dimension, figsize=(6, 6 * feature_dimension))
    for i in range(feature_dimension): # make 1d plots of 1d interactions
        posterior = opt_posterior.predict_additive_component(xplot1d, D, [i])
        mean, std = posterior.mean(), posterior.stddev()
        ax[i].plot(xplot1d[:,i], mean, color="blue", label="additive")
        ax[i].fill_between(xplot1d[:,i], mean - 2 * std,mean + 2 * std, alpha=0.2, color="blue")
        ax[i].set_title(f"$f_{i}(x_{i})$" if not sobol else f"$f_{i}(x_{i})$ has sobol ${opt_posterior.get_sobol_index(D, [i]):.2f}$")
        truth = lof[i](xplot1d[:,i])
        ax[i].plot(xplot1d[:,i], truth, color="black", label="truth")
        ax[i].legend()
        ax[i].scatter(Xtr[:,i], ytr, c="red", marker="x", label="data")
    # pprint(opt_posterior.__dict__)

def plot_posterior_with_components2d(opt_posterior, active_dimensions: list[int] = [0, 1], axis_to_plot = 0):
        if len(active_dimensions) not in [1,2]:
            raise ValueError("Only 1 or 2 active dimensions are supported")
        contour_plot(opt_posterior.predict_additive_component(
                mesh2d(active_dimensions=active_dimensions),
                D,
                [0, 1]
        ).mean(), ax[axis_to_plot])

def generate_additive_plots(
        list_of_functions: List = [],
        num_points: int = 50,
        type_of_points: str = 'normal',
        standard_deviation: float = 1.0,
        bound_on_drawn_points: List[float] = [-5, 5],
        noise_level: float = 0.1,
        train_test_split_value: float = 0.2,
        num_points_to_plot: int = 100,
        bounds: List[float] = [-5, 5],
        scale_X: bool = False,
        scale_y: bool = False,
        key = jr.PRNGKey(0),
        kernel_type: str = 'RBF',
        trainable_noise: bool = True,
        max_interaction_depth: int = 1,
        optimiser: ox.GradientTransformation = ox.adam,
        num_iters: int = 100,
        save_file: bool = False,
        save_file_name: str = None,
        path: str = None,
        ):
    """Generate additive plots for a list of functions."""

    def f(x):
        return sum([list_of_functions[i](x[:,i:i+1]) for i in range(len(list_of_functions))])
    feature_dimension = len(list_of_functions)

    if type_of_points not in ['normal', 'uniform']:
        raise ValueError('type_of_points should be either normal or uniform')
    if kernel_type not in ['RBF', 'OrthogonalRBF']:
        raise ValueError('kernel should be either RBF or OrthogonalRBF')
    
    if type_of_points == 'normal':
        X = jr.normal(key, (num_points, feature_dimension))*standard_deviation
    else:
        X = jr.uniform(key, (num_points, feature_dimension), minval=bound_on_drawn_points[0], maxval=bound_on_drawn_points[1])

    y = f(X) + jr.normal(key, (num_points, 1)) * noise_level
    if train_test_split_value:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=train_test_split_value, random_state=42)
    else:
        Xtr, Xte, ytr, yte = X, X, y, y

    if scale_X:
        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)
    if scale_y:
        scaler = StandardScaler().fit(ytr)
        ytr = scaler.transform(ytr)
        yte = scaler.transform(yte)

    xplot1d = jnp.stack([jnp.linspace(bounds[0], bounds[1], num_points_to_plot) for _ in range(feature_dimension)]).T


    D = gpx.Dataset(Xtr, ytr)
    meanf = gpx.mean_functions.Zero()
    if kernel_type == 'RBF':
        base_kernels = [gpx.kernels.RBF(active_dims=[i], lengthscale=jnp.array([1.0])) for i in range(feature_dimension)]
    else:
        base_kernels = [OrthogonalRBF(active_dims = [i], lengthscale=jnp.array([1.0])) for i in range(feature_dimension)]
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n, obs_stddev=noise_level)
    if not trainable_noise:
        likelihood = likelihood.replace_trainable(obs_stddev=False)
    obj = gpx.objectives.ConjugateLOOCV(negative=True)
    kernel = AdditiveKernel(
        kernels=base_kernels,
        interaction_variances=jnp.array([1.0]*(max_interaction_depth + 1)) * jnp.var(D.y), 
        max_interaction_depth=max_interaction_depth, 
        )
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    posterior = AdditiveConjugatePosterior(prior =prior, likelihood=likelihood)
    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=obj,
        train_data=D,
        optim=optimiser,
        num_iters=num_iters,
        key=key, 
        safe=False,
        verbose=False)
    plt.figure()
    plt.plot(history)
    plt.title(save_file_name)
    plt.savefig(path + "/" + save_file_name + "LIKELIHOODGRAPH" + ".png")
    
    if save_file: 
        file_path = path + "/" + save_file_name + ".pkl"
        with open(file_path, "wb") as file:
            pickle.dump(opt_posterior.__dict__, file)
    pprint(opt_posterior.__dict__)
    

    def plot_posterior_with_components_here(opt_posterior, sobol=True):
        fig, ax = plt.subplots(ncols=1, nrows=feature_dimension, figsize=(6, 6 * feature_dimension))
        for i in range(feature_dimension): # make 1d plots of 1d interactions
            posterior = opt_posterior.predict_additive_component(xplot1d, D, [i])
            mean, std = posterior.mean(), posterior.stddev()
            ax[i].plot(xplot1d[:,i], mean, color="blue", label="additive")
            ax[i].fill_between(xplot1d[:,i], mean - 2 * std,mean + 2 * std, alpha=0.2, color="blue")
            ax[i].set_title(f"$f_{i+1}(x_{i+1})$" if not sobol else f"$f_{i+1}(x_{i+1})$ has sobol ${opt_posterior.get_sobol_index(D, [i]):.2f}$")
            truth = list_of_functions[i](xplot1d[:,i])
            ax[i].plot(xplot1d[:,i], truth, color="black", label="truth")
            ax[i].legend()
            ax[i].scatter(Xtr[:,i], ytr, c="red", marker="x", label="data")
        if save_file:
            fig.savefig(path + "/" + save_file_name + ".png")
    

    plot_posterior_with_components_here(opt_posterior)
    plt.show()









