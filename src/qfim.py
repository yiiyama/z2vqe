"""Generic functions for QFIM calculation."""
import numpy as np
import jax
import jax.numpy as jnp


def make_ansatz_layer_fn(generators):
    diagonals = np.diagonal(generators, axis1=1, axis2=2)
    diagmat = np.zeros_like(generators)
    dim = generators.shape[1]
    diagmat[:, np.arange(dim), np.arange(dim)] = diagonals
    is_diagonal = np.all(np.isclose(generators, diagmat), axis=(1, 2))

    @jax.jit
    def fn(params, state):
        for igen, generator in enumerate(generators):
            if is_diagonal[igen]:
                state *= jnp.exp(-0.5j * params[igen] * diagonals[igen])
            else:
                state = jax.scipy.linalg.expm(-0.5j * params[igen] * generator) @ state

        return state

    return fn


def make_ansatz_fn(generators, num_layers):
    layer_fn = make_ansatz_layer_fn(generators)

    @jax.jit
    def apply_layer(ilayer, args):
        params, state = args
        layer_params = params.reshape((-1, generators.shape[0]))[ilayer]
        state = layer_fn(layer_params, state)
        return (params, state)

    @jax.jit
    def fn(params, initial_state):
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
