"""Z2 LGT."""
import numpy as np
import scipy
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Pauli


def calculate_num_params(num_sites, num_layers, trans_inv):
    """Calculate the number of parameters for the given system configuration.

    Copied from the VQE notebook.
    """
    if trans_inv:
        return 5 * num_layers
    return 3 * num_layers * num_sites


def initial_state(num_sites, boundary_cond):
    """Return a QuantumCircuit corresponding to initial-state prep plus the number of qubits.

    Copied from the VQE notebook.
    """
    if boundary_cond == 'closed':
        num_qubits = 2 * num_sites
    else:
        num_qubits = 2 * num_sites - 1

    qc = QuantumCircuit(num_qubits)

    # Xゲートを最初に適用し、次にHゲートを適用する
    for i in range(3, num_qubits, 4):
        qc.x(i)
    for i in range(0, num_qubits, 2):
        qc.h(i)

    return qc, num_qubits


def initial_state_vector(num_sites):
    """Compute the initial state as a state vector."""
    num_qubits = 2 * num_sites
    init_state = np.zeros((2,) * num_qubits)
    init_state[(0,) * num_qubits] = 1.
    paulix = np.array([[0., 1.], [1., 0.]])
    hadamard = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2.)
    for iq in range(0, num_qubits, 4):
        init_state = np.moveaxis(np.tensordot(paulix, init_state, [1, iq]), 0, iq)
    for iq in range(1, num_qubits, 2):
        init_state = np.moveaxis(np.tensordot(hadamard, init_state, [1, iq]), 0, iq)

    return init_state.reshape(-1)


def xyz_gate(angle):
    """Make a QuantumCircuit corresponding to the hopping term Hamiltonian unitary.

    Copied from the VQE notebook.
    """
    circuit = QuantumCircuit(3, name='xyz_gate')
    circuit.cx(2, 0)
    circuit.h(1)
    circuit.rz(np.pi / 2, 2)
    circuit.cx(2, 1)
    circuit.ry(-1 * angle, 2)
    circuit.cx(0, 2)
    circuit.ry(angle, 2)
    circuit.cx(0, 2)
    circuit.cx(2, 1)
    circuit.h(1)
    circuit.rz(-np.pi / 2, 2)
    circuit.cx(2, 0)
    return circuit


def z2_ansatz(num_sites, num_layers, num_params, boundary_cond, trans_inv):
    """Make a QuantumCircuit corresponding to initial-state prep + ansatz unitary.

    Copied from the VQE notebook and refactored with a layer-wise function call.
    """
    circuit, _ = initial_state(num_sites, boundary_cond)
    theta = ParameterVector('theta', length=num_params)

    ip = 0
    if trans_inv:
        for _ in range(num_layers):
            circuit.compose(z2_ansatz_layer(num_sites, boundary_cond, theta[ip:ip + 5]),
                            inplace=True)
            ip += 5

    return circuit


def z2_ansatz_layer(num_sites, boundary_cond, params=None):
    """Make a QuantumCircuit corresponding to a single ansatz layer unitary.

    Copied from the VQE notebook.
    """
    if boundary_cond == 'closed':
        num_qubits = 2 * num_sites
    else:
        num_qubits = 2 * num_sites - 1

    if params is None:
        params = ParameterVector('theta', 5)

    circuit = QuantumCircuit(num_qubits)
    ip = 0
    for i in range(1, num_qubits - 2, 4):
        circuit.append(xyz_gate(params[ip]), [i + 2, i + 1, i])
    ip += 1

    for i in range(3, num_qubits - 2, 4):
        circuit.append(xyz_gate(params[ip]), [i + 2, i + 1, i])
    if num_sites % 2 == 0 and boundary_cond == 'closed':
        circuit.append(xyz_gate(params[ip]), [1, 0, num_qubits - 1])
    ip += 1

    circuit.barrier()

    for i in range(3, num_qubits, 4):
        circuit.rz(params[ip], i)
    ip += 1
    for i in range(1, num_qubits, 4):
        circuit.rz(params[ip], i)
    ip += 1

    for i in range(0, num_qubits, 2):
        circuit.rx(params[ip], i)
    ip += 1

    return circuit


def create_hamiltonian(num_sites, j_hopping, f_gauge, mass, overall_coeff, boundary_cond,
                       overall_coeff_cond):
    """Create a SparsePauliOp for the Hamiltonian.

    Copied from the VQE notebook.
    """
    num_qubits = 2 * num_sites if boundary_cond == 'closed' else 2 * num_sites - 1

    def op_positions(op1, pos1, op2=None, pos2=None, op3=None, pos3=None):
        product = [Pauli('I')] * num_qubits
        product[pos1] = op1
        if op2 is not None:
            product[pos2] = op2
        if op3 is not None and pos3 < num_qubits:
            product[pos3] = op3
        return SparsePauliOp.from_list([("".join(map(str, product)), 1)])

    ops = []
    for n in range(num_sites - 1):
        pos1, pos2, pos3 = (2 * n) % num_qubits, (2 * n + 1) % num_qubits, (2 * n + 2) % num_qubits
        ops += [(-j_hopping / 2) * op_positions(Pauli('X'), pos1, Pauli('Z'), pos2, Pauli('X'),
                                                pos3),
                (-j_hopping / 2) * op_positions(Pauli('Y'), pos1, Pauli('Z'), pos2, Pauli('Y'),
                                                pos3),
                -f_gauge * op_positions(Pauli('X'), pos2) if pos2 < num_qubits - 1 else None,
                (mass / 2 * (-1)**n) * op_positions(Pauli('Z'), pos1)]
        if overall_coeff_cond:
            ops += [(-(overall_coeff * (-1)**n) * op_positions(Pauli('X'), pos1 - 1, Pauli('Z'),
                                                               pos1, Pauli('X'), pos2))]

    if boundary_cond == 'closed':
        ops += [(-j_hopping / 2) * op_positions(Pauli('X'), num_qubits - 2, Pauli('Z'),
                                                num_qubits - 1, Pauli('X'), 0),
                (-j_hopping / 2) * op_positions(Pauli('Y'), num_qubits - 2, Pauli('Z'),
                                                num_qubits - 1, Pauli('Y'), 0),
                -f_gauge * op_positions(Pauli('X'), -1),
                (mass / 2 * (-1)) * op_positions(Pauli('Z'), num_qubits - 2)]
        if overall_coeff_cond:
            ops += [(overall_coeff * op_positions(Pauli('X'), num_qubits - 3, Pauli('Z'),
                                                  num_qubits - 2, Pauli('X'), num_qubits - 1))]

    return sum(ops).simplify()


def get_symmetry_sector(state: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the Gauss's law + U1 charge symmetry sector of a state vector.

    The Gauss's law sector is given as an array of X[i-1,i]Z[i]X[i,i+1] eigenvalues. The U1 sector
    is given as a mean of Z[i] eigenvalues.

    Args:
        state: State vector.

    Returns:
        A tuple of an array for the Gauss's law sector and a float for the U1 sector.
    """

    num_qubits = np.log2(state.shape[0]).astype(int)
    num_sites = num_qubits // 2
    # Reshape the state to make it applicable to single-qubit operations
    state = state.reshape((2,) * num_qubits)
    # Single-qubit Hadamard matrix
    hadamard = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2.)
    # Transform all link qubits to X basis
    for isite in range(num_sites):
        # tensordot places the resulting qubit to axis 0, so we move it back to the original
        # position with moveaxis
        state = np.moveaxis(np.tensordot(hadamard, state, [1, isite * 2 + 1]), 0, isite * 2 + 1)

    # Compute the Z eigenvalue of each qubit
    zeigvals = np.empty(num_qubits, dtype=int)
    for iqubit in range(num_qubits):
        # Move the qubit to axis 0 (just to make array indexing easier)
        moved = np.moveaxis(state, iqubit, 0)
        # If the amplitude of moved[0] is 0, the state has only |1> for this qubit -> Z=-1
        if np.allclose(moved[0], 0.):
            zeigvals[iqubit] = -1
        # If the amplitude of moved[1] is 0, the state has only |0> for this qubit -> Z=1
        elif np.allclose(moved[1], 0.):
            zeigvals[iqubit] = 1
        else:
            raise ValueError('Not a Gauss law eigenstate')

    # gauss_sector[i] = zeigvals[2i-1] * zeigvals[2i] * zeigvals[2i+1]
    gauss_sector = np.roll(zeigvals, -1)[::2] * zeigvals[::2] * np.roll(zeigvals, 1)[::2]
    # u1_sector[i] = sum_i [zeigvals[2i] / N]
    u1_sector = np.sum(zeigvals[::2]) / num_sites
    return gauss_sector, u1_sector


def subspace_diagonalization(
    hamiltonian: SparsePauliOp,
    gauss_sector: np.ndarray,
    u1_sector: float
) -> tuple[np.ndarray, np.ndarray]:
    """Project the Hamiltonian to the given symmetry sector and diagonalize.

    Args:
        hamiltonian: Hamiltonian given as a SparsePauliOp.
        gauss_sector: Array of XZX eigenvalues.
        u1_sector: Mean of site Z eigenvalues.

    Returns:
        Arrays returned by np.linalg.eigh.
    """
    num_qubits = hamiltonian.num_qubits
    num_sites = num_qubits // 2

    # Enumerate all binary indices from 0 to 2^Nq - 1
    # state_binary = [[0, ..., 0, 0], [0, ..., 0, 1], [0, ..., 1, 0], ..., [1, ..., 1, 1]]
    state_binary = (np.arange(2 ** num_qubits)[:, None] >> np.arange(num_qubits)[None, ::-1]) % 2
    # Convert the binary indices to Z eigenvalues
    state_zeigvals = 1 - 2 * state_binary
    # Compute the products of Z eigenvalues of neighboring three qubits
    # state_gauss = [[1, 1, ..., 1, 1], [-1, 1, ..., -1, -1], [1, 1, ..., -1, -1], ...]
    state_gauss = (np.roll(state_zeigvals, -1, axis=1)[:, ::2]
                   * state_zeigvals[:, ::2]
                   * np.roll(state_zeigvals, 1, axis=1)[:, ::2])
    # Compute the site (even qubit) sum of Z eigenvalues
    # state_u1 = [Ns, Ns, Ns-2, ..., -Ns]
    state_u1 = np.sum(state_zeigvals[:, ::2], axis=1)
    # Extract the indices (symmetry subspace) that match the given sectors
    subspace = np.nonzero(
        np.all(np.equal(state_gauss, gauss_sector[None, :]), axis=1)
        & np.equal(state_u1, np.round(u1_sector * num_sites).astype(int))
    )[0]
    subdim = subspace.shape[0]

    # Make the projectors onto the obtained subspace
    # First create an array of one-hot vectors (s_0, s_1, ...) where s_i is a column vector with 1
    # at subspace[i] and 0 elsewhere
    subbasis = np.zeros((2 ** num_qubits, subdim))
    subbasis[subspace, np.arange(subdim)] = 1.
    # The current array corresponds to a projector in the Z basis. Since the subspace is defined in
    # terms of X[link]Z[site]X[link] eigenstates, we transform the link qubits to X basis.
    subbasis = subbasis.reshape((2,) * num_qubits + (subdim,))
    hadamard = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2.)
    for isite in range(num_sites):
        subbasis = np.moveaxis(
            np.tensordot(hadamard, subbasis, [1, 2 * isite + 1]),
            0,
            2 * isite + 1
        )
    # Convert the projector array to CSR
    subbasis = scipy.sparse.csr_array(subbasis.reshape((2 ** num_qubits, subdim)))

    # Project the Hamiltonian to the subspace
    block_hamiltonian = (subbasis.T @ (hamiltonian.to_matrix(sparse=True) @ subbasis)).todense()
    # Clean up machine-epsilon entries
    block_hamiltonian = np.where(np.isclose(block_hamiltonian, 0.), 0., block_hamiltonian)

    return np.linalg.eigh(block_hamiltonian)
