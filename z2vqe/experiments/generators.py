"""Construct and project the generators."""
from functools import partial
import numpy as np
from scipy.sparse import csc_array
import jax
import jax.numpy as jnp
from fastdla.generators.z2lgt_physical_hva import (
    z2lgt_physical_hva_generators,
    z2lgt_physical_symmetry_eigenspace,
    z2lgt_physical_u1_charges
)
from z2vqe.experiments.configurations import CONFIGS


def _clean_array(arr, tol=1.e-12, npmod=np):
    res = npmod.where(npmod.isclose(arr.real, 0., atol=tol), 0., arr.real).astype(arr.dtype)
    if arr.dtype == np.complex128:
        res += 1.j * npmod.where(npmod.isclose(arr.imag, 0., atol=tol), 0., arr.imag)
    return res


_clean_array_jit = jax.jit(partial(_clean_array, npmod=jnp))


def clean_array(arr, tol=1.e-12, npmod=np):
    if npmod is np:
        return _clean_array(arr, tol=tol)
    else:
        return _clean_array_jit(arr, tol=tol)


@jax.jit
def validate_symmetry(gen, subspace):
    """Check commutation between a Hermitian generator and a subspace projector."""
    pg = subspace @ (subspace.conjugate().T @ gen)
    return jnp.allclose(pg - pg.conjugate().T, 0.)


def get_generators(config, gauss_eigvals):
    combinations, qnums = CONFIGS[config]
    genops = z2lgt_physical_hva_generators(gauss_eigvals, gauge_op='Z')
    gen_spmat = genops.to_matrices(sparse=True)

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

    shape_csr = (u1_eigenidx.shape[0], gen_spmat[0].shape[1])
    shape_csc = (u1_eigenidx.shape[0], u1_eigenidx.shape[0])
    gen_spmat = [project_sparse(csc_array(project_sparse(mat, shape_csr)), shape_csc)
                 for mat in gen_spmat]
    gen_mat = jnp.concatenate([mat.todense()[None, ...] for mat in gen_spmat])
    # Take the negative imaginary part because our Z2 LGT generators are pure imaginary
    gen_mat = -gen_mat.imag

    gen_mat = jnp.stack([jnp.sum(gen_mat[np.array(comb)], axis=0) for comb in combinations])
    ngen = gen_mat.shape[0]
    print('generators:', ngen)

    subspace = z2lgt_physical_symmetry_eigenspace(gauss_eigvals, npmod=jnp, **qnums)
    subspace = subspace[u1_eigenidx]
    subdim = subspace.shape[1]
    print('subspace:', subdim)
    print('symmetries:', [bool(validate_symmetry(gen, subspace)) for gen in gen_mat])

    gen_mat = jnp.einsum('ji,gjk,kl->gil', subspace.conjugate(), gen_mat, subspace)
    gen_mat = clean_array(gen_mat, npmod=jnp)

    trace = jnp.trace(gen_mat, axis1=1, axis2=2)
    print('trace:', trace)
    if jnp.allclose(trace, 0., atol=1.e-11):
        norm_eye = jnp.eye(subdim, dtype=np.complex128) / subdim
        gen_mat -= jnp.tile(norm_eye[None, ...], (ngen, 1, 1)) * trace[:, None, None]

    return gen_mat, u1_eigenidx, subspace


if __name__ == '__main__':
    import os
    from argparse import ArgumentParser
    from pathlib import Path
    import logging
    import h5py

    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('num_fermions', type=int)
    parser.add_argument('--gpu')
    parser.add_argument('--out-dir')
    parser.add_argument('--log-level', default='warning')
    options = parser.parse_args()

    logging.basicConfig(level=getattr(logging, options.log_level.upper()))
    if options.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu
    jax.config.update('jax_enable_x64', True)

    generators, u1idx, proj = get_generators(options.config, [1, -1] * options.num_fermions)

    out_dir = Path(options.out_dir or '.')
    filename = f'generators-{options.config}-{options.num_fermions}.h5'
    with h5py.File(out_dir / filename, 'w', libver='latest') as out:
        out.create_dataset('hva_gen_proj', data=generators)
        out.create_dataset('u1_eigenidx', data=u1idx)
        out.create_dataset('subspace', data=proj)
