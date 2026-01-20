"""Generic functions for QFIM calculation and VQE."""
import time
import logging
import numpy as np
import jax
import jax.numpy as jnp
import jaxopt

LOG = logging.getLogger(__name__)


def make_ansatz_layer_fn(generators):
    evals, evecs = jnp.linalg.eigh(-1.j * generators)

    @jax.jit
    def fn(params, state):
        for igen in range(generators.shape[0]):
            state = evecs[igen].conjugate().T @ state
            state *= jnp.exp(-0.5j * params[igen] * evals[igen])
            state = evecs[igen] @ state

        return state

    return fn


def make_ansatz_fn(generators, num_layers):
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


def make_qfim_fn(generators, num_layers):
    ansatz_fn = make_ansatz_fn(generators, num_layers)
    jacobian_fn = jax.jit(jax.jacfwd(ansatz_fn))

    @jax.jit
    def fn(params, initial_state):
        state = ansatz_fn(params, initial_state)
        # pylint: disable-next=not-callable
        jacobian = jacobian_fn(params, initial_state)
        qfim = jnp.tensordot(jacobian.conjugate(), jacobian, axes=(0, 0)).real
        qfim -= jnp.outer(
            jnp.tensordot(jacobian.conjugate(), state, axes=(0, 0)),
            jnp.tensordot(state.conjugate(), jacobian, axes=(0, 0))
        ).real
        return qfim

    return fn


def make_cost_fn(generators, num_layers):
    ansatz_fn = make_ansatz_fn(generators, num_layers)

    @jax.jit
    def fn(params, initial_state, hamiltonian):
        state = ansatz_fn(params, initial_state)
        energy = (state.conjugate() @ hamiltonian @ state).real
        return energy

    return fn


def vqe(
    cost_fn,
    initial_state,
    hamiltonian,
    param_init,
    maxiter,
    tol=1.e-4,
    target=-np.inf,
    stepsize=0.,
    print_every=100
):
    solver = jaxopt.GradientDescent(fun=cost_fn, stepsize=stepsize, acceleration=False)
    update = jax.pmap(
        jax.jit(
            jax.vmap(solver.update, in_axes=(0, 0, None, None))
        ),
        in_axes=(0, 0, None, None)
    )
    value = jax.pmap(
        jax.jit(
            jax.vmap(cost_fn, in_axes=(0, None, None))
        ),
        in_axes=(0, None, None)
    )

    num_params = param_init.shape[-1]
    num_instances = np.prod(param_init.shape[:-1])

    params = jnp.array(param_init)
    state = jax.pmap(jax.jit(jax.vmap(solver.init_state)))(params)

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
