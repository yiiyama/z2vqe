"""Generic functions for QFIM calculation and VQE."""
from collections.abc import Callable
import time
import logging
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
import jaxopt

LOG = logging.getLogger(__name__)


def make_ansatz_layer_fn(generators: jax.Array) -> Callable:
    """Return a function that applies one ansatz layer to a state vector.

    Args:
        generators: Hermitian generator matrices.

    Returns:
        Ansatz layer function.
    """
    evals, evecs = jnp.linalg.eigh(generators)

    @jax.jit
    def fn(params, state):
        extra_dims = tuple(range(1, state.ndim))
        for igen in range(generators.shape[0]):
            state = evecs[igen].conjugate().T @ state
            state = jnp.expand_dims(jnp.exp(1.j * params[igen] * evals[igen]), extra_dims) * state
            state = evecs[igen] @ state

        return state

    return fn


def make_ansatz_fn(generators: jax.Array, num_layers: int) -> Callable:
    """Return a function that applies the ansatz unitary to a state vector.

    Args:
        generators: Hermitian generator matrices.
        num_layers: Number of ansatz layers.

    Returns:
        Ansatz function.
    """
    layer_fn = make_ansatz_layer_fn(generators)

    @jax.jit
    def apply_layer(ilayer, args):
        params, state = args
        state = layer_fn(params[ilayer], state)
        return (params, state)

    @jax.jit
    def fn(params, initial_state):
        params = params.reshape((num_layers, generators.shape[0]))
        _, state = jax.lax.fori_loop(0, num_layers, apply_layer, (params, initial_state))
        return state

    return fn


def make_qfim_fn(generators: jax.Array, num_layers: int) -> Callable:
    """Return a function that computes the QFIM of the parametrized state.

    Args:
        generators: Hermitian generator matrices.
        num_layers: Number of ansatz layers.

    Returns:
        QFIM function.
    """
    ansatz_fn = make_ansatz_fn(generators, num_layers)
    jacobian_fn = jax.jit(jax.jacfwd(ansatz_fn))

    @jax.jit
    def fn(params, initial_state):
        psi = ansatz_fn(params, initial_state)
        # pylint: disable-next=not-callable
        dpsi = jacobian_fn(params, initial_state)
        # <ψ|dψ> = <ψ_0|U† (-iUH~j)|ψ_0> = -i <ψ_0|H~j|ψ_0>  (H~j = U[0:j]† Hj U[0:j])
        mim_psidpsi = (psi.conjugate() @ dpsi).imag
        qfim = jnp.tensordot(dpsi.conjugate(), dpsi, axes=(0, 0)).real
        qfim -= jnp.outer(mim_psidpsi, mim_psidpsi)
        return qfim

    return fn


def make_cost_fn(generators: jax.Array, num_layers: int) -> Callable:
    """Return a function that computes the energy expectation value of a parametrized state.

    Args:
        generators: Hermitian generator matrices.
        num_layers: Number of ansatz layers.

    Returns:
        Energy function.
    """
    ansatz_fn = make_ansatz_fn(generators, num_layers)

    @jax.jit
    def fn(params, initial_state, hamiltonian):
        state = ansatz_fn(params, initial_state)
        energy = (state.conjugate() @ hamiltonian @ state).real
        return energy

    return fn


def vqe(
    generators: jax.Array,
    num_layers: int,
    initial_state: jax.Array,
    hamiltonian: jax.Array,
    instances_per_device: int = 1,
    param_init: Optional[np.ndarray] = None,
    maxiter: int = 10_000,
    tol: float = 1.e-4,
    target: float = -np.inf,
    stepsize: float = 0.,
    print_every: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Perform VQE with an HVA.

    Args:
        generators: Hermitian generator matrices.
        num_layers: Number of ansatz layers.
        initial_state: Initial state.
        hamiltonian: Target Hamiltonian.
        instances_per_device: Number of parallel VQE instances to run per device.
        param_init: Initial parameter values.
        maxiter: Maximum gradient descent iterations.
        tol: Stopping criterion (Cutoff of the maximum of the changes in parameters between GD
            iterations.)
        target: Stopping criterion (Energy threshold)
        stepsize: Step size for GD.
        print_every: Verbosity setting.

    Returns:
        Loss curve and the parameter history.
    """
    rng = np.random.default_rng()
    cost_fn = make_cost_fn(generators, num_layers)

    solver = jaxopt.GradientDescent(fun=cost_fn, stepsize=stepsize, acceleration=False)
    update = jax.jit(jax.vmap(solver.update, in_axes=(0, 0, None, None)))
    value = jax.jit(jax.vmap(cost_fn, in_axes=(0, None, None)))
    init_state = jax.jit(jax.vmap(solver.init_state))
    num_params = generators.shape[0] * num_layers
    num_dev = jax.device_count()
    if num_dev > 1:
        update = jax.pmap(update, in_axes=(0, 0, None, None))
        value = jax.pmap(value, in_axes=(0, None, None))
        init_state = jax.pmap(init_state)
        default_param_shape = (num_dev, instances_per_device, num_params)
    else:
        default_param_shape = (instances_per_device, num_params)

    if param_init is None:
        param_init = rng.normal(0., 2. * np.pi, size=default_param_shape)
    num_instances = instances_per_device * num_dev

    params = jnp.array(param_init)
    state = init_state(params)

    energies = np.empty((num_instances, maxiter + 1))
    parameters = np.empty((num_instances, maxiter + 1, num_params))

    start_time = time.time()
    LOG.info('Compiling the cost function..')
    energies[:, 0] = value(params, initial_state, hamiltonian).reshape(num_instances)
    update(params, state, initial_state, hamiltonian)  # pylint: disable=not-callable
    LOG.info('Compilation of the cost function took %.2f seconds', time.time() - start_time)

    parameters[:, 0, :] = params.reshape(num_instances, num_params)

    start_time = time.time()
    for istep in range(maxiter):
        params, state = update(params, state, initial_state, hamiltonian)
        energy = value(params, initial_state, hamiltonian).reshape(num_instances)
        energies[:, istep + 1] = energy
        parameters[:, istep + 1, :] = params.reshape(num_instances, num_params)
        if print_every > 0 and istep % print_every == 0:
            LOG.info('Iteration: %d, elapsed time: %.2f seconds', istep, time.time() - start_time)
        if energy < target or np.max(np.abs(np.diff(parameters[:, istep:istep + 2], axis=1))) < tol:
            break

    return energies[:, :istep], parameters[:, :istep]


def qfim_saturation(
    generators: jax.Array,
    initial_state: jax.Array,
    points_per_device: int = 1,
    param_init_fn: Optional[Callable[[int], np.ndarray]] = None,
    tol: float = 1.e-10,
    return_svals: bool = False,
    mode: str = 'binary_search',
    **search_params
):
    """Find the maximum rank of the QFIM for the HVA.

    Args:
        generators: Hermitian generator matrices.
        initial_state: Initial state.
        points_per_device: Number of parameter points to compute per device.
        param_init_fn: Function returning initial parameter values given the number of parameters.
        tol: Singular value cutoff.
        return_svals: Whether to return the singular values themselves in addition to the ranks.
        mode: Search mode.
    """
    rng = np.random.default_rng()
    num_dev = jax.device_count()

    if param_init_fn is None:
        def param_init_fn(num_params):
            if num_dev > 1:
                shape = (num_dev, points_per_device, num_params)
            else:
                shape = (points_per_device, num_params)
            return rng.normal(0., 2. * np.pi, size=shape)

    def compute_qfim_svals(num_layer):
        LOG.info('Computing QFIM rank for ansatz with %d layers', num_layer)
        qfim_fn = jax.jit(
            jax.vmap(
                make_qfim_fn(generators, num_layer),
                in_axes=(0, None)
            )
        )
        if num_dev > 1:
            qfim_fn = jax.pmap(qfim_fn, in_axes=(0, None))

        num_params = num_layer * generators.shape[0]
        params = param_init_fn(num_params)
        matrices = qfim_fn(params, initial_state).reshape((-1, num_params, num_params))
        return np.linalg.svd(matrices, compute_uv=False, hermitian=True)

    def rank(svals):
        return np.count_nonzero(svals > tol, axis=1)

    def compute_qfim_rank(num_layer):
        svals = compute_qfim_svals(num_layer)
        return rank(svals)

    result_fn = compute_qfim_svals if return_svals else compute_qfim_rank

    if mode == 'binary_search':
        LOG.info('Searching for maximum QFIM rank..')
        num_layer = search_params.get('initial_step', 16)
        results = {num_layer: result_fn(num_layer)}
        if return_svals:
            ranks = {num_layer: rank(results[num_layer])}
        else:
            ranks = results
        initial_step = num_layer
        rsat = None
        while True:
            num_layer += initial_step
            results[num_layer] = result_fn(num_layer)
            if return_svals:
                ranks[num_layer] = rank(results[num_layer])
            if np.isclose(np.mean(ranks[num_layer]), np.mean(ranks[num_layer - initial_step])):
                rsat = np.mean(ranks[num_layer])
                LOG.info('Found saturation rank %f', rsat)
                num_layer -= 2 * initial_step
                break

        increments = []
        incr = initial_step
        while True:
            incr //= 2
            increments.append(incr)
            if incr == 1:
                break

        LOG.info('Moving with increments %s', increments)
        for increment in increments:
            while True:
                num_layer += increment
                if num_layer in results:
                    break
                results[num_layer] = result_fn(num_layer)
                if return_svals:
                    ranks[num_layer] = rank(results[num_layer])
                if np.isclose(np.mean(ranks[num_layer]), rsat):
                    num_layer -= increment
                    break

        num_layers = sorted(results.keys())
        ranks = np.array([ranks[nl] for nl in num_layers])

    elif mode == 'scan':
        num_layer = search_params.get('nl_init', 1)
        num_layers = [num_layer]
        results = [result_fn(num_layer)]
        if return_svals:
            ranks = [rank(results[0])]
        else:
            ranks = results
        while True:
            num_layer += search_params.get('increment', 1)
            num_layers.append(num_layer)
            results.append(result_fn(num_layer))
            if return_svals:
                ranks.append(rank(results[-1]))
            if np.isclose(np.mean(ranks[-1]), np.mean(ranks[-2])):
                break

        ranks = np.array(ranks)

    retval = (ranks, np.array(num_layers))
    if return_svals:
        retval += (results,)
    return retval
