"""Generic functions for QFIM calculation and VQE."""
from collections.abc import Callable
import time
import logging
from functools import partial
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


def make_qfim_fn(
    generators: jax.Array,
    num_layers: int,
    vmap: bool = False,
    pmap: bool = False
) -> Callable:
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

    if vmap:
        fn = jax.jit(jax.vmap(fn, in_axes=(0, None)))
    if pmap:
        fn = jax.pmap(fn, in_axes=(0, None))

    return fn


def make_cost_fn(
    generators: jax.Array,
    num_layers: int,
    vmap: bool = False,
    pmap: bool = False
) -> Callable:
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

    if vmap:
        fn = jax.jit(jax.vmap(fn, in_axes=(0, None, None)))
    if pmap:
        fn = jax.pmap(fn, in_axes=(0, None, None))

    return fn


def make_curvature_fn(
    cost_fn: Callable,
    vmap: bool = False,
    pmap: bool = False
) -> Callable:
    """Return a function that computes the eigenvalues of the Hessian of the cost function."""
    hess_fn = jax.hessian(cost_fn)

    @jax.jit
    def fn(params, initial_state, hamiltonian):
        hess = hess_fn(params, initial_state, hamiltonian)
        return jnp.linalg.eigvalsh(hess)

    if vmap:
        fn = jax.jit(jax.vmap(fn, in_axes=(0, None, None)))
    if pmap:
        fn = jax.pmap(fn, in_axes=(0, None, None))

    return fn


def vqe(
    generators: jax.Array,
    num_layers: int,
    initial_state: jax.Array,
    hamiltonian: jax.Array,
    instances_per_device: int = 1,
    params: Optional[np.ndarray] = None,
    maxiter: int = 10_000,
    stepsize: float = 0.,
    acceleration: bool = True,
    tol: float = 1.e-4,
    target: float = -np.inf,
    solve_only: bool = False,
    print_every: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    cost_fn = make_cost_fn(generators, num_layers, vmap=True, pmap=jax.device_count() > 1)
    if params is None:
        params = random_params(generators.shape[0] * num_layers, instances_per_device)

    return run_vqe(cost_fn, initial_state, hamiltonian, params, maxiter,
                   stepsize=stepsize, acceleration=acceleration,
                   tol=tol, target=target, solve_only=solve_only, print_every=print_every)


def random_params(num_params, instances_per_device):
    rng = np.random.default_rng()
    if (num_dev := jax.device_count()) > 1:
        shape = (num_dev, instances_per_device, num_params)
    else:
        shape = (instances_per_device, num_params)
    return rng.normal(0., 2. * np.pi, size=shape)


def run_vqe(
    cost_fn,
    initial_state,
    hamiltonian,
    params,
    maxiter,
    stepsize=0.,
    acceleration=True,
    tol: float = 1.e-4,
    target: float = -np.inf,
    return_curvature: bool = False,
    solve_only: bool = False,
    print_every: int = 100
):
    cost_fn_s = cost_fn
    while hasattr(cost_fn_s, '__wrapped__'):
        cost_fn_s = cost_fn_s.__wrapped__
    solver = jaxopt.GradientDescent(fun=cost_fn_s, stepsize=stepsize, acceleration=acceleration)
    update_fn = jax.jit(jax.vmap(solver.update, in_axes=(0, 0, None, None)))
    init_state_fn = jax.jit(jax.vmap(solver.init_state))
    if return_curvature:
        curvature_fn = make_curvature_fn(cost_fn_s, vmap=True)

    if jax.device_count() > 1:
        update_fn = jax.pmap(update_fn, in_axes=(0, 0, None, None))
        init_state_fn = jax.pmap(init_state_fn)
        if return_curvature:
            curvature_fn = jax.pmap(curvature_fn, in_axes=(0, None, None))

    num_instances = np.prod(params.shape[:-1])
    num_params = params.shape[-1]
    params = jnp.array(params)
    state = init_state_fn(params)

    energies = np.empty((num_instances, maxiter + 1))
    parameters = np.empty((num_instances, maxiter + 1, num_params))
    stepsizes = np.empty((num_instances, maxiter))
    if return_curvature:
        curvatures = np.empty((num_instances, maxiter + 1, num_params))

    if not solve_only:
        LOG.info('Computing initial loss values')
        energies[:, 0] = cost_fn(params, initial_state, hamiltonian).reshape(num_instances)
        parameters[:, 0, :] = params.reshape(num_instances, num_params)
        if return_curvature:
            curvature = curvature_fn(params, initial_state, hamiltonian)
            curvatures[:, 0, :] = curvature.reshape((num_instances, num_params))

    LOG.info('Minimization start')
    start_time = time.time()
    for istep in range(1, maxiter + 1):
        params, state = update_fn(params, state, initial_state, hamiltonian)
        stepsizes[:, istep - 1] = state.stepsize.reshape(num_instances)
        if not solve_only:
            energy = cost_fn(params, initial_state, hamiltonian).reshape(num_instances)
            energies[:, istep] = energy
            parameters[:, istep, :] = params.reshape(num_instances, num_params)
            if return_curvature:
                curvature = curvature_fn(params, initial_state, hamiltonian)
                curvatures[:, istep, :] = curvature.reshape((num_instances, num_params))
        if print_every > 0 and istep % print_every == 0:
            LOG.info('Iteration: %d, elapsed time: %.2f seconds', istep, time.time() - start_time)
        if tol > 0. and np.max(np.abs(np.diff(parameters[:, istep - 1:istep + 1], axis=1))) < tol:
            break
        if not solve_only and np.max(energy) < target:
            break

    retval = (energies[:, :istep + 1], parameters[:, :istep + 1], stepsizes)
    if return_curvature:
        return retval + (curvatures,)
    return retval


def compute_qfim_svals(
    generators: jax.Array,
    initial_state: jax.Array,
    num_layer: int,
    param_init_fn: Optional[Callable[[int], np.ndarray]] = None,
    points_per_device: int = 1
):
    LOG.info('Computing QFIM rank for ansatz with %d layers', num_layer)
    rng = np.random.default_rng()
    num_dev = jax.device_count()

    if param_init_fn is None:
        def param_init_fn(num_params):
            if num_dev > 1:
                shape = (num_dev, points_per_device, num_params)
            else:
                shape = (points_per_device, num_params)
            return rng.normal(0., 2. * np.pi, size=shape)

    qfim_fn = make_qfim_fn(generators, num_layer, vmap=True, pmap=num_dev > 1)
    num_params = num_layer * generators.shape[0]
    params = param_init_fn(num_params)
    matrices = qfim_fn(params, initial_state).reshape((-1, num_params, num_params))
    return np.linalg.svd(matrices, compute_uv=False, hermitian=True)


def qfim_saturation(
    generators: jax.Array,
    initial_state: jax.Array,
    points_per_device: int = 1,
    param_init_fn: Optional[Callable[[int], np.ndarray]] = None,
    tol: float = 1.e-10,
    rtol: float = 1.e-6,
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
    def rank(svals):
        absolute = svals > tol
        relative = np.concatenate(
            [np.ones((svals.shape[0], 1), dtype=bool), svals[:, 1:] > rtol * svals[:, :-1]],
            axis=1
        )
        return np.count_nonzero(absolute & relative, axis=1)

    get_svals = partial(compute_qfim_svals, generators, initial_state,
                        param_init_fn=param_init_fn, points_per_device=points_per_device)

    if mode == 'binary_search':
        LOG.info('Searching for maximum QFIM rank..')
        num_layer = search_params.get('initial_step', 16)
        svals = {num_layer: get_svals(num_layer)}
        ranks = {num_layer: rank(svals[num_layer])}
        initial_step = num_layer
        rsat = None
        while True:
            num_layer += initial_step
            svals[num_layer] = get_svals(num_layer)
            ranks[num_layer] = rank(svals[num_layer])
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
            while num_layer + increment not in svals:
                num_layer += increment
                svals[num_layer] = get_svals(num_layer)
                ranks[num_layer] = rank(svals[num_layer])

                if np.isclose(np.mean(ranks[num_layer]), rsat):
                    num_layer -= increment
                    break

        num_layers = sorted(svals.keys())
        ranks = np.array([ranks[nl] for nl in num_layers])

    elif mode == 'scan':
        num_layer = search_params.get('nl_init', 1)
        num_layers = [num_layer]
        svals = [get_svals(num_layer)]
        ranks = [rank(svals[0])]
        while True:
            num_layer += search_params.get('increment', 1)
            num_layers.append(num_layer)
            svals.append(get_svals(num_layer))
            ranks.append(rank(svals[-1]))
            if np.isclose(np.mean(ranks[-1]), np.mean(ranks[-2])):
                break

        ranks = np.array(ranks)

    retval = (ranks, np.array(num_layers))
    if return_svals:
        retval += (svals,)
    return retval
