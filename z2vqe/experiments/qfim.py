"""Compute the QFIM rank."""
import os
from pathlib import Path
import logging
import numpy as np
import h5py
import jax
from z2vqe.qfim_vqe import qfim_saturation


def compute_qfim_rank(generators, subspace, points_per_device):
    initial_state = np.zeros(subspace.shape[0], dtype=np.complex128)
    initial_state[0] = 1.
    initial_state = subspace.conjugate().T @ initial_state

    ranks = qfim_saturation(generators, initial_state, points_per_device)
    print('QFIM:', ranks[-1])
    return ranks


def main(config, num_fermions, out_dir, log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    jax.config.update('jax_enable_x64', True)

    out_dir = Path(out_dir or '.')
    filename = f'generators-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'r', libver='latest') as source:
        hva_gens = -1.j * source['hva_gen_proj'][()]
        subspace_proj = source['subspace'][()]

    qfim_ranks = compute_qfim_rank(hva_gens, subspace_proj, 100)

    filename = f'qfim-{config}-{num_fermions}.h5'
    with h5py.File(out_dir / filename, 'w', libver='latest') as out:
        out.create_dataset('ranks', data=qfim_ranks)


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
