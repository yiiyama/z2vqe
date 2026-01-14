"""Z2 LGT with link DOF only."""
import numpy as np
import jax
import jax.numpy as jnp
from fastdla.sparse_pauli_sum import SparsePauliSumArray, SparsePauliSum
from fastdla.linalg.eigenspace import get_eigenspace
from fastdla.generators.spin_chain import translation_eigenspace
from fastdla.linalg.block_diagonalization import sbd_fast


def mass_term_even(gauss_eigvals):
    num_sites = len(gauss_eigvals)
    paulis = ['Z' + 'I' * (num_sites - 2) + 'Z']
    paulis += ['I' * (num_sites - ilink - 2) + 'ZZ' + 'I' * ilink
               for ilink in range(1, num_sites - 1, 2)]
    return SparsePauliSum(paulis, np.asarray(gauss_eigvals[::2]))


def mass_term_odd(gauss_eigvals):
    num_sites = len(gauss_eigvals)
    paulis = ['I' * (num_sites - ilink - 2) + 'ZZ' + 'I' * ilink
              for ilink in range(0, num_sites, 2)]
    return SparsePauliSum(paulis, -np.asarray(gauss_eigvals[1::2]))


def hopping_term_even(gauss_eigvals):
    num_sites = len(gauss_eigvals)
    paulis = ['I' * (num_sites - ilink - 1) + 'X' + 'I' * ilink for ilink in range(0, num_sites, 2)]
    paulis.append('Z' + 'I' * (num_sites - 3) + 'ZX')
    paulis += ['I' * (num_sites - ilink - 2) + 'ZXZ' + 'I' * (ilink - 1)
               for ilink in range(2, num_sites, 2)]
    gauss_eigvals = np.asarray(gauss_eigvals)
    zxz_coeff = -0.5 * gauss_eigvals[::2] * gauss_eigvals[1::2]
    return SparsePauliSum(paulis, np.concatenate([np.ones(num_sites // 2) * 0.5, zxz_coeff]))


def hopping_term_odd(gauss_eigvals):
    num_sites = len(gauss_eigvals)
    paulis = ['I' * (num_sites - ilink - 1) + 'X' + 'I' * ilink for ilink in range(1, num_sites, 2)]
    paulis += ['I' * (num_sites - ilink - 2) + 'ZXZ' + 'I' * (ilink - 1)
               for ilink in range(1, num_sites - 1, 2)]
    paulis.append('XZ' + 'I' * (num_sites - 3) + 'Z')
    gauss_eigvals = np.asarray(gauss_eigvals)
    zxz_coeff = -0.5 * gauss_eigvals[::2] * gauss_eigvals[1::2]
    return SparsePauliSum(paulis, np.concatenate([np.ones(num_sites // 2) * 0.5, zxz_coeff]))


def gauge_term(gauss_eigvals):
    num_sites = len(gauss_eigvals)
    paulis = ['I' * (num_sites - ilink - 1) + 'Z' + 'I' * ilink for ilink in range(num_sites)]
    return SparsePauliSum(paulis, np.ones(num_sites))


def hamiltonian(gauss_eigvals, mass, coupling):
    return (gauge_term(gauss_eigvals)
            + mass * (mass_term_even(gauss_eigvals) + mass_term_odd(gauss_eigvals))
            + coupling * (hopping_term_even(gauss_eigvals) + hopping_term_odd(gauss_eigvals)))


def generators(gauss_eigvals):
    return SparsePauliSumArray([
        hopping_term_even(gauss_eigvals),
        hopping_term_odd(gauss_eigvals),
        mass_term_even(gauss_eigvals),
        mass_term_odd(gauss_eigvals),
        gauge_term(gauss_eigvals)
    ])


def symmetry_sector(
    gauss_eigvals,
    u1_charge=0,
    t2_momentum=0,
    cp_parity=1,
    diagonalize=True,
    npmod=np
):
    num_sites = len(gauss_eigvals)
    subspace = None

    if u1_charge is not None:
        idx = npmod.arange(2 ** num_sites)
        bidx = ((idx[:, None] >> npmod.arange(num_sites)[None, ::-1]) % 2).astype(bool)
        charges = npmod.zeros(2 ** num_sites, dtype=int)
        for ilink, gn in enumerate(gauss_eigvals):
            parity = bidx[:, num_sites - 1 - ilink] == bidx[:, (-ilink) % num_sites]
            charges += (-1 + 2 * parity.astype(int)) * gn
        idx = np.nonzero(charges == u1_charge)[0]
        if npmod is np:
            subspace = np.zeros((2 ** num_sites, idx.shape[0]), dtype=np.complex128)
            subspace[idx, np.arange(idx.shape[0])] = 1.
        else:
            subspace = jax.nn.one_hot(idx, 2 ** num_sites).T

    if t2_momentum is not None:
        subspace = translation_eigenspace(t2_momentum, basis=subspace, num_spins=num_sites,
                                          shift=2, npmod=npmod)

    if cp_parity is not None:
        def cp_kernel(basis):
            idx = npmod.arange(2 ** num_sites)
            bidx = (idx[:, None] >> npmod.arange(num_sites)[None, ::-1]) % 2
            dest_bidx = npmod.roll(bidx[:, ::-1], 1, axis=1)
            dest_idx = npmod.sum(dest_bidx * (1 << npmod.arange(num_sites)[::-1]), axis=1)
            return basis[dest_idx, :] - cp_parity * basis

        if npmod is jnp:
            cp_kernel = jax.jit(cp_kernel)

        subspace = get_eigenspace(cp_kernel, basis=subspace)

    if diagonalize:
        genmat = generators(gauss_eigvals).to_matrices()
        proj_gen = npmod.einsum('ij,gik,kl->gjl', subspace.conjugate(), genmat[3:], subspace)
        diagonalizer = sbd_fast(proj_gen, hermitian=True, npmod=npmod)
        subspace = subspace @ diagonalizer

    return subspace
