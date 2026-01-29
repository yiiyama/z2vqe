"""Compute the DLA of the HVA generators."""
import os
from pathlib import Path
import logging
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from fastdla.lie_closure import lie_closure


def compute_dla(generators_herm):
    shard_basis = jax.device_count() > 1
    dla = lie_closure(jnp.array(-1.j * generators_herm), skew_hermitian=True,
                      shard_basis=shard_basis)
    if shard_basis:
        basis, indices = dla
        dla = np.array(basis)[indices]

    print('DLA:', dla.shape[0])
    return dla


def compute_block_dla(num_f, tr, blocks, subspace, u1_eigenidx):
    eigvecs = np.empty((2 ** (2 * num_f), tr.shape[1]), dtype=np.complex128)
    eigvecs[u1_eigenidx] = subspace @ tr
    # Which column has the state |00..0> ?
    col = np.argmax(np.abs(eigvecs[0]))
    iblock = np.searchsorted(np.cumsum([block.shape[1] for block in blocks]), col, side='right')
    print('Block:', blocks[iblock].shape[1])
    dla = lie_closure(-1.j * jnp.array(blocks[iblock]), skew_hermitian=True)
    print('Block DLA:', dla.shape[0])
    return dla


def main(config, num_fermions, out_dir, log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    jax.config.update('jax_enable_x64', True)

    out_dir = Path(out_dir or '.')
    filename = f'generators-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        generators_herm = source['hva_gen_proj'][()]
        subspace = source['subspace'][()]
        u1_eigenidx = source['u1_eigenidx'][()]

    dla = compute_dla(generators_herm)

    block_dla = None
    filename = f'sbd-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        if 'block1' in source:
            tr = source['transform'][()]
            blocks = [source[f'block{i}'][()] for i in range(len(source.keys()) - 1)]
            block_dla = compute_block_dla(num_fermions, tr, blocks, subspace, u1_eigenidx)

    filename = f'dla-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'w', libver='latest') as out:
        out.create_dataset('dla', data=dla)
        if block_dla is not None:
            out.create_dataset('block_dla', data=block_dla)


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
