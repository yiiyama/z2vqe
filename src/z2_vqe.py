# pylint: disable=import-outside-toplevel, wrong-import-position, invalid-name
"""Functions for Z2 LGT VQE."""
import time
from datetime import datetime
from numbers import Number
import numpy as np
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
import jaxopt
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.parametervector import ParameterVectorElement


def make_circuit_fn_elements(ansatz):
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


def make_circuit_fn(*args):
    """Return a function (params, state) ↦ state."""
    from qujax.statetensor import _gate_func_to_unitary, apply_gate

    if len(args) == 1 and isinstance(args[0], QuantumCircuit):
        gates_seq, qubit_inds_seq, param_inds_seq = make_circuit_fn_elements(args[0])
    else:
        gates_seq, qubit_inds_seq, param_inds_seq = args

    def circuit_fn(params, statetensor):
        params = jnp.atleast_1d(params)

        for gate_func, qubit_inds, param_inds in zip(
            gates_seq, qubit_inds_seq, param_inds_seq
        ):
            gate_unitary = _gate_func_to_unitary(
                gate_func, qubit_inds, param_inds, params
            )
            statetensor = apply_gate(statetensor, gate_unitary, qubit_inds)
        return statetensor

    return circuit_fn


def make_static_state(circuit):
    """Return a state constructed by a non-parametrized QuantumCircuit."""
    from qujax.statetensor import all_zeros_statetensor

    return make_circuit_fn(circuit)(jnp.array([]),
                                    all_zeros_statetensor(circuit.num_qubits))


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

    return jax.jit(get_statetensor_to_expectation_func(pauli_prods, supports, hamiltonian.coeffs))


def make_ansatz_circuit_fn(ansatz_layer, num_layers):
    """Return a circuit function from repeated ansatz layers."""
    if isinstance(ansatz_layer, list):
        assert len(ansatz_layer) == num_layers
        layer_fns = [make_circuit_fn(layer) for layer in ansatz_layer]

        @jax.jit
        def ansatz_circuit_fn(params, state):
            for ilayer, layer_fn in enumerate(layer_fns):
                layer_params = jax.lax.dynamic_slice(params, (ilayer * 5,), (5,))
                state = layer_fn(layer_params, state)
            return state

    else:
        ansatz_layer_fn = jax.jit(make_circuit_fn(ansatz_layer))

        def _ansatz_layer(ilayer, val):
            params, state = val
            layer_params = jax.lax.dynamic_slice(params, (ilayer * 5,), (5,))
            # pylint: disable-next=not-callable
            return params, ansatz_layer_fn(layer_params, state)

        @jax.jit
        def ansatz_circuit_fn(params, state):
            _, state = jax.lax.fori_loop(0, num_layers, _ansatz_layer, (params, state))
            return state

    return ansatz_circuit_fn


def make_state_fn(init_state, ansatz_layer, num_layers):
    ansatz_fn = make_ansatz_circuit_fn(ansatz_layer, num_layers)
    initial_state = make_static_state(init_state)
    return jax.jit(lambda params: ansatz_fn(params, initial_state))


def make_cost_fn(init_state, ansatz_layer, num_layers, hamiltonian):
    state_fn = make_state_fn(init_state, ansatz_layer, num_layers)
    expval_fn = make_expval_fn(hamiltonian)
    # pylint: disable-next=not-callable
    return jax.jit(lambda params: expval_fn(state_fn(params)).real)


def make_qfim_fn(init_state, ansatz_layer, num_layers):
    state_fn = make_state_fn(init_state, ansatz_layer, num_layers)

    def statevector_fn(params):
        # pylint: disable-next=not-callable
        return state_fn(params).reshape(-1)

    jacobian_fn = jax.jit(jax.jacfwd(statevector_fn))

    @jax.jit
    def qfim_fn(params):
        state = statevector_fn(params)
        # pylint: disable-next=not-callable
        jacobian = jacobian_fn(params)
        qfim = jnp.tensordot(jacobian.conjugate(), jacobian, axes=(0, 0))
        qfim -= jnp.outer(
            jnp.tensordot(jacobian.conjugate(), state, axes=(0, 0)),
            jnp.tensordot(state.conjugate(), jacobian, axes=(0, 0))
        )
        return 4. * qfim.real

    return qfim_fn


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


def vqe_jaxopt(cost_fn, init, maxiter, stepsize=0., print_every=100):
    solver = jaxopt.GradientDescent(fun=cost_fn, stepsize=stepsize, acceleration=False)
    update = jax.pmap(jax.jit(jax.vmap(solver.update)))
    value = jax.pmap(jax.jit(jax.vmap(cost_fn)))

    num_params = init.shape[-1]
    num_instances = np.prod(init.shape[:-1])

    params = jnp.array(init)
    state = jax.pmap(jax.jit(jax.vmap(solver.init_state)))(params)

    energies = np.empty((num_instances, maxiter + 1))
    parameters = np.empty((num_instances, maxiter + 1, num_params))

    start_time = time.time()
    print('Compiling the cost function..')
    energies[:, 0] = value(params).reshape(num_instances)
    update(params, state)  # pylint: disable=not-callable
    print(f'Compilation of the cost function took {time.time() - start_time} seconds')

    parameters[:, 0, :] = params.reshape(num_instances, num_params)

    start_time = time.time()
    for istep in range(maxiter):
        params, state = update(params, state)
        energies[:, istep + 1] = value(params).reshape(num_instances)
        parameters[:, istep + 1, :] = params.reshape(num_instances, num_params)
        if print_every > 0 and istep % print_every == 0:
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Iteration: {istep}'
                  f', elapsed time: {time.time() - start_time} seconds')

    return energies, parameters
