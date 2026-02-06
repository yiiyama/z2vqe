"""Construct and project the generators."""

import os
from pathlib import Path
import logging
from functools import partial
import numpy as np
from scipy.sparse import csc_array
import h5py
import jax
import jax.numpy as jnp
from fastdla.generators.z2lgt_physical_hva import (
    z2lgt_physical_hva_generators,
    z2lgt_physical_symmetry_eigenspace,
    z2lgt_physical_u1_charges
)
from fastdla.algorithms.gram_schmidt import gram_schmidt
from z2vqe.experiments.configurations import CONFIGS, MASS, COUPLING
from z2vqe.experiments.tools import clean_array


@jax.jit
def validate_symmetry(gen, subspace):
    """Check commutation between a real symmetric generator and a real symmetric projector."""
    pg = subspace @ (subspace.T @ gen)
    return jnp.allclose(pg, pg.T)


def project_su2_rep(num_fermions, u1_eigenidx, subspace, hgen_mat):
    gs = partial(gram_schmidt, cutoff=1.e-6, npmod=jnp)
    init_state = np.zeros(2 ** (2 * num_fermions))
    init_state[[0, -1]] = np.sqrt(0.5)
    vectors = (subspace.T @ init_state[u1_eigenidx])[None, :]
    basis = jnp.zeros((subspace.shape[1],) * 2)
    basis, basis_size = gs(vectors, basis=basis, basis_size=0)
    while True:
        vectors = (hgen_mat[1] @ basis[:basis_size].T).T
        basis, new_size = gs(vectors, basis=basis, basis_size=basis_size)
        vectors = (hgen_mat[0] @ basis[:new_size].T).T
        basis, new_size = gs(vectors, basis=basis, basis_size=new_size)
        if new_size == basis_size:
            break
        basis_size = new_size

    basis = basis[:basis_size]
    return subspace @ basis.T, basis @ hgen_mat @ basis.T


def get_generators(config, gauss_eigvals):
    combinations, qnums = CONFIGS[config]
    ops_elements = z2lgt_physical_hva_generators(gauss_eigvals, gauge_op='Z')
    spmats_elements = ops_elements.to_matrices(sparse=True)

    charges = z2lgt_physical_u1_charges(gauss_eigvals, npmod=np)
    u1_eigenidx = np.nonzero(charges == 0)[0]

    def project_sparse(spmat, shape):
        data = spmat.data
        indices = spmat.indices
        indptr = spmat.indptr
        data = np.concatenate([data[indptr[row]:indptr[row + 1]] for row in u1_eigenidx])
        indices = np.concatenate([indices[indptr[row]:indptr[row + 1]] for row in u1_eigenidx])
        selected_lows = indptr[u1_eigenidx]
        selected_highs = indptr[u1_eigenidx + 1]
        indptr = np.concatenate([[0], np.cumsum(selected_highs - selected_lows)])
        return type(spmat)((data, indices, indptr), shape=shape)

    shape_csr = (u1_eigenidx.shape[0], spmats_elements[0].shape[1])
    shape_csc = (u1_eigenidx.shape[0], u1_eigenidx.shape[0])
    spmats_elements = [project_sparse(csc_array(project_sparse(mat, shape_csr)), shape_csc)
                       for mat in spmats_elements]
    # Take the negative imaginary part because our Z2 LGT generators are pure imaginary
    gen_elements = jnp.concatenate([-mat.todense()[None, ...].imag for mat in spmats_elements])

    subspace = z2lgt_physical_symmetry_eigenspace(gauss_eigvals, npmod=jnp, **qnums).real
    subspace = subspace[u1_eigenidx]
    subdim = subspace.shape[1]
    print('subspace:', subdim)

    gen_mats = jnp.stack([jnp.sum(gen_elements[np.array(comb)], axis=0) for comb in combinations])
    ngen = gen_mats.shape[0]
    print('generators:', ngen)
    print('symmetries:', [bool(validate_symmetry(gen, subspace)) for gen in gen_mats])
    gen_mats = subspace.T @ gen_mats @ subspace

    if config == 'm_h':
        nf = len(gauss_eigvals) // 2
        subspace, gen_mats = project_su2_rep(nf, u1_eigenidx, subspace, gen_mats)
        subdim = subspace.shape[1]
        print('su2 rep:', subdim)

    gen_mats = clean_array(gen_mats, npmod=jnp)

    trace = jnp.trace(gen_mats, axis1=1, axis2=2)
    print('trace:', trace)
    if jnp.allclose(trace, 0., atol=1.e-11):
        norm_eye = jnp.eye(subdim, dtype=gen_mats.dtype) / subdim
        gen_mats -= jnp.tile(norm_eye[None, ...], (ngen, 1, 1)) * trace[:, None, None]

    subspace_full = np.zeros((2 ** len(gauss_eigvals), subspace.shape[1]))
    subspace_full[u1_eigenidx] = subspace

    hamiltonian = jnp.sum(gen_elements[0:2], axis=0)
    hamiltonian += MASS * np.sum(gen_elements[2:4], axis=0)
    hamiltonian += COUPLING * np.sum(gen_elements[4:], axis=0)
    hamiltonian = subspace.T @ hamiltonian @ subspace
    hamiltonian = clean_array(hamiltonian, npmod=jnp)

    return gen_mats, subspace_full, hamiltonian


def main(config, num_fermions, out_dir, log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    jax.config.update('jax_enable_x64', True)

    gen_mats, subspace, hamiltonian = get_generators(config, [1, -1] * num_fermions)

    out_dir = Path(out_dir or '.')
    filename = f'generators-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'w', libver='latest') as out:
        out.create_dataset('gen_mats', data=gen_mats)
        out.create_dataset('subspace', data=subspace)
        out.create_dataset('hamiltonian', data=hamiltonian)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('num_fermions', type=int)
    parser.add_argument('--gpu')
    parser.add_argument('--out-dir')
    parser.add_argument('--log-level', default='warning')
    options = parser.parse_args()

    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

    main(options.config, options.num_fermions, options.out_dir, options.log_level)
