# pylint: disable=import-outside-toplevel, wrong-import-position, invalid-name
"""Functions for Z2 LGT VQE."""
import time
from numbers import Number
import numpy as np
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
import jaxopt
from qiskit.circuit.parametervector import ParameterVectorElement


def make_state_fn_elements(ansatz):
    import qujax

    constant_gates = {
        'cx': lambda: qujax.gates.CX,
        'x': lambda: qujax.gates.X,
        'h': lambda: qujax.gates.H
    }

    def make_norm_gate(gate):
        return lambda p: gate(p / np.pi)

    def make_norminv_gate(gate):
        return lambda p: gate(-p / np.pi)

    pinorm_gates = {gate: make_norm_gate(gate)
                    for gate in [qujax.gates.Rx, qujax.gates.Ry, qujax.gates.Rz]}
    inv_gates = {gate: make_norminv_gate(gate)
                 for gate in [qujax.gates.Rx, qujax.gates.Ry, qujax.gates.Rz]}

    def arr_to_callable(arr):
        return lambda: arr

    null_arr = jnp.array([])

    gates_seq = []
    qubit_inds_seq = []
    param_inds_seq = []
    for datum in ansatz.decompose(['xyz_gate']).data:
        if datum.operation.name == 'barrier':
            continue

        if datum.params:
            match datum.operation.name:
                case 'rx':
                    gate = qujax.gates.Rx
                case 'ry':
                    gate = qujax.gates.Ry
                case 'rz':
                    gate = qujax.gates.Rz

            if isinstance(datum.params[0], Number):
                gate = arr_to_callable(
                    gate(datum.params[0] / np.pi)
                )
                param_inds_seq.append(null_arr)
            elif isinstance(datum.params[0], ParameterVectorElement):
                gate = pinorm_gates[gate]
                param_inds_seq.append(jnp.array([p.index for p in datum.params]))
            else:
                gate = inv_gates[gate]
                param_inds_seq.append(
                    jnp.array([list(p.parameters)[0].index for p in datum.params])
                )

            gates_seq.append(gate)
        else:
            gates_seq.append(constant_gates[datum.operation.name])
            param_inds_seq.append(null_arr)

        qubit_inds_seq.append([ansatz.find_bit(q).index for q in datum.qubits])

    return gates_seq, qubit_inds_seq, param_inds_seq


def make_state_fn(*args):
    from qujax.statetensor import _gate_func_to_unitary, apply_gate

    if len(args) == 1:
        gates_seq, qubit_inds_seq, param_inds_seq = make_state_fn_elements(args[0])
    else:
        gates_seq, qubit_inds_seq, param_inds_seq = args

    def state_fn(params, statetensor):
        params = jnp.atleast_1d(params)

        for gate_func, qubit_inds, param_inds in zip(
            gates_seq, qubit_inds_seq, param_inds_seq
        ):
            gate_unitary = _gate_func_to_unitary(
                gate_func, qubit_inds, param_inds, params
            )
            statetensor = apply_gate(statetensor, gate_unitary, qubit_inds)
        return statetensor

    return state_fn


def make_expval_fn(hamiltonian):
    from qujax_dev.statetensor_observable import get_statetensor_to_expectation_func

    pauli_prods = []
    supports = []
    for pauli in hamiltonian.paulis:
        pauli_prod = []
        support = []
        for iq, op in enumerate(pauli.to_label()[::-1]):
            if op != 'I':
                pauli_prod.append(op)
                support.append(iq)

        pauli_prods.append(pauli_prod)
        supports.append(support)

    return get_statetensor_to_expectation_func(pauli_prods, supports, hamiltonian.coeffs)


def make_ansatz_fn(ansatz_layer, num_layers):
    ansatz_layer_fn = jax.jit(make_state_fn(ansatz_layer))

    def _ansatz_layer(ilayer, val):
        params, state = val
        layer_params = jax.lax.dynamic_slice(params, (ilayer * 5,), (5,))
        # pylint: disable-next=not-callable
        return params, ansatz_layer_fn(layer_params, state)

    def ansatz_fn(params, state):
        _, state = jax.lax.fori_loop(0, num_layers, _ansatz_layer, (params, state))
        return state

    return ansatz_fn


def make_cost_fn(init_state, ansatz_layer, num_layers, hamiltonian):
    from qujax.statetensor import all_zeros_statetensor

    ansatz_fn = make_ansatz_fn(ansatz_layer, num_layers)
    expval_fn = make_expval_fn(hamiltonian)
    initial_state = make_state_fn(init_state)(jnp.array([]),
                                              all_zeros_statetensor(init_state.num_qubits))
    return lambda params: expval_fn(ansatz_fn(params / np.pi, initial_state)).real


def vqe_scipy(cost_fn, init, maxiter):
    start_time = time.time()
    print('Compiling the cost function..')
    cost_fn(init)
    print(f'Compilation of the cost function took {time.time() - start_time} seconds')

    energies = []

    def callback(xk):
        energy = cost_fn(xk)
        energies.append(energy)
        if len(energies) % 10 == 1:
            print(f'Iteration: {len(energies)}, elapsed time: {time.time() - start_time} seconds')

    start_time = time.time()
    result = minimize(cost_fn, init, method="COBYLA", options={'maxiter': maxiter},
                      callback=callback, tol=1e-13)
    print(f"Completed {len(energies)} iterations with {result.nfev + len(energies)} function"
          f" evaluations in {time.time() - start_time} seconds")

    return np.array(energies)


def vqe_jaxopt(cost_fn, init, maxiter, print_every=100):
    solver = jaxopt.GradientDescent(fun=cost_fn, acceleration=False)
    init_state_fn = jax.pmap(jax.vmap(solver.init_state))
    update = jax.pmap(jax.vmap(solver.update))
    value = jax.pmap(jax.vmap(cost_fn))

    params = jnp.array(init)
    state = init_state_fn(params)

    start_time = time.time()

    print('Compiling the cost function..')
    update(params, state)  # pylint: disable=not-callable
    print(f'Compilation of the cost function took {time.time() - start_time} seconds')

    start_time = time.time()
    energies = []
    for _ in range(maxiter):
        params, state = update(params, state)
        energies.append(value(params).reshape(-1))
        if print_every > 0 and len(energies) % print_every == 1:
            print(f'Iteration: {len(energies)}, elapsed time: {time.time() - start_time} seconds')

    return np.array(energies)
